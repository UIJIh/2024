# paper: https://aclanthology.org/2021.emnlp-main.552/
# reference implementation: https://github.com/princeton-nlp/SimCSE
#
# this implementation only supports Unsup-SimCSE.
# if you want to run the training of Sup-SimCSE, please modify this code yourself.

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from classopt import classopt
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from sts import STSEvaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# classopt is a library for parsing command line arguments in a dataclass style.
# different from argparse, classopt can enjoy the benefits of type hints.
# see: https://github.com/moisutsu/classopt (let's star it!)
@classopt(default_long=True)
class Args:
    model_name: str = "bert-base-uncased" # uncased; it does not make a difference between english and English.
    # any data set in line-by-line text format can be used.
    # however, it is worth noting that diversity of the dataset is important for SimCSE.
    # see: https://github.com/princeton-nlp/SimCSE/issues/62
    dataset_dir: Path = "./datasets/unsup-simcse"
    sts_dir: Path = "./datasets/sts"
    output_dir: Path = "./outputs"

    # for more detailed hyperparameter settings, see Appendix.A of the paper
    # FYI: SimCSE is not sensitive to batch sizes and learning rates
    batch_size: int = 64
    # the number of epochs is 1 for Unsup-SimCSE, and 3 for Sup-SimCSE in the paper
    epochs: int = 1
    lr: float = 3e-5
    # num_warmup_steps is 0 by default (i.e. no warmup)
    num_warmup_steps: int = 0

    # see Table D.1 of the paper
    temperature: float = 0.05

    # FYI: max_seq_len of reference implementation is 32
    # it seems short, but it is enough for the STS task (Semantic Textual Similarity 문장끼리 유사한지)
    # you should be careful when you apply SimCSE to other tasks that require longer sequences to be handled properly.
    # for other hyperparameters, see Appendix.A of the paper.
    max_seq_len: int = 32

    # FYI: the paper says that the evaluation interval is 250 steps.
    # however, the example training script of official implementation uses 125 steps.
    # this does not seem to be a problem when the number of training steps is large (i.e. batch size is small), as in BERT (batch_size=64),
    # but it may make some difference when the number of steps is small (i.e. batch size is large), as in RoBERTa (batch_size=512).
    # see: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/run_unsup_example.sh
    eval_logging_interval: int = 250

    # if you want to use `fp16`, you may encounter some issues.
    # see: https://github.com/princeton-nlp/SimCSE/issues/38#issuecomment-855457923
    device: str = "cuda:0"

    # due to various influences such as implementation and hardware, the same random seed does not always produce the same results.
    # the hyperparameters used in the paper are tuned with a single random seed,
    # so the results may be slightly different from the paper.
    # if you train your own model, you should preferably re-tune the hyperparameters.
    # FYI: https://github.com/princeton-nlp/SimCSE/issues/63
    seed: int = 42


# Reading text line by line is a very simple processing, so we don't need to use a Dataset class actually.
# However we define a dedicated class for future extensibility.
@dataclass # from dataclasses import dataclass
class SimCSEDataset(Dataset):
    path: Path
    data: List[str] = None

    # For simplicity, this dataset class is designed to tokenize text for each loop,
    # but if performance is more important, you should tokenize all text in advance.
    def __post_init__(self):
        self.data = []
        with self.path.open() as f:
            # to prevent whole text into memory at once
            for line in f:
                line = line.strip() # 앞 뒤 공백 제거
                if line: # 줄이 비어있지 않으면
                    self.data.append(line)

    def __getitem__(self, index: int) -> Tensor: # index를 받아서 tensor를 반환
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class SimCSEModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__() # nn.Module
        # you can use any models
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # define additional MLP layer 
        # see Section 6.3 of the paper for more details
        # refenrece: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/simcse/models.py#L19
        self.hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        # RoBERTa variants don't have token_type_ids, so this argument is optional
        token_type_ids: Tensor = None,
    ) -> Tensor:
        # shape of input_ids: (batch_size, seq_len)
        # shape of attention_mask: (batch_size, seq_len)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # take representations of [CLS] token
        # we only implement the best performing pooling, [CLS], for simplicity
        # you can easily extend to other poolings (such as mean pooling or max pooling) by edting this line
        # shape of last_hidden_state: (batch_size, seq_len, hidden_size)
        emb = outputs.last_hidden_state[:, 0] # 각 seq의 0번째 토큰인 CLS token의 representation을 추출
                                              # sequence에서 1개만 추출하니까 (batch_size, hidden_size)

        # original SimCSE uses MLP layer only during training
        # see: Table 6 of the paper
        # this trick is a bit complicated, so you may omit it when training your own model
        if self.training:
            emb = self.dense(emb)
            emb = self.activation(emb)
        # shape of emb: (batch_size, hidden_size)
        return emb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: Args):
    logging.set_verbosity_error()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name) # bert-base-uncased
    model: SimCSEModel = SimCSEModel(args.model_name).to(args.device) # bert-base-uncased

    train_dataset = SimCSEDataset(args.dataset_dir / "train.txt") # Path 객체는 이런식으로 전달한다고 함

    # `collate_fn` is for processing the list of samples to form a batch
    # see: https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
    def collate_fn(batch: List[str]) -> BatchEncoding:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

    # see: https://pytorch.org/docs/stable/data.html
    #      https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn, # 이걸 해주면 패딩/토큰화 등등 사용자 정의 처리 가능, 자동으로 각 배치에 대해 DataLoader가 처리한다고 함
        batch_size=args.batch_size, # dataset중에 이 배치 사이즈만큼의 list가 collate_fn에 전달돼서 tokenizer 수행
        shuffle=True,
        # num_workers and pin_memory are for speeding up training
        num_workers=4,
        pin_memory=True,
        # batch_size varies in the last batch because
        # the last batch size will be the number of remaining samples (i.e. len(train_dataloader) % batch_size)
        # to avoid unstablity of contrastive learning, we drop the last batch
        drop_last=True,
    )

    # FYI: huggingface/transformers' AdamW implementation is deprecated and you should use PyTorch's AdamW instead.
    # see: https://github.com/huggingface/transformers/issues/3407
    #      https://github.com/huggingface/transformers/issues/18757
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # reference implementation uses a linear scheduler with warmup, which is a default scheduler of transformers' Trainer
    # with num_training_steps = 0 (i.e. no warmup)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, # 0 (no warmup)
        # len(train_dataloader) is the number of steps in one epoch
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    # evaluation class for STS task
    # we use a simple cosine similarity as a semantic similarity
    # and use Spearman's correlation as an evaluation metric
    # see: `sts.py`
    sts = STSEvaluation(sts_dir=args.sts_dir)

    # encode sentences (List[str]) and output embeddings (Tensor)
    # this is for evaluation (no_grad)
    @torch.inference_mode() # encode 함수가 호출될 때 torch.inference_mode()를 공통적으로 작동시킴 (유지보수)
    def encode(texts: List[str]) -> torch.Tensor: # (len(texts),)
        embs = []
        model.eval()
        for text in chunked(texts, args.batch_size * 8): # text를 args.batch_size * 8 이만큼 나누기
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # SimCSE uses MLP layer only during training
            # in this implementation, we use `model.training` to switch between training and evaluation
            emb = model(**batch.to(args.device)) # SimCSEModel(bert-base-uncased) return -> cls token, representation!
                                                 # (batch=text, hidden)
            embs.append(emb.cpu()) # [(args.batch_size * 8, hidden), (args.batch_size * 8, hidden), ...] 
            '''
            만약에 texts 리스트에 1000개의 문장이 있고, batch size가 16이라면,
            1000개 문장을 128(16*8)개씩 묶어서 나눔 
            그러면 뭉텅이가 약 8개가 나올거임 (args.batch_size * 8, hidden) 이 텐서가 texts//args.batch_size * 8 (8)개만큼 list에 담겨있음
            args.batch_size * 8 개의 text에 대해서 8번 embs를 얻은 것임!!!!!!!!!!! (맨 마지막은 개수가 다를수도)
            이렇게 뭉텅이 나눈걸 Tesor로 합침! 
            >> selbst_text.ipynb 참고
            '''
        # shape of output: (len(texts), hidden_size) 
        return torch.cat(embs, dim=0) # 어쨌든 texts를 여러 chunk들로 나눈걸 다시 textx 전체만큼 합치겠단거임

    # evaluation before training
    model.eval()
    best_stsb = sts.dev(encode=encode) # 임베딩 변환해서 그걸로 sts eval 해봄
    best_step = 0

    # evaluate the model and store metrics before training
    # this is important to check the appropriateness of training procedure
    print(f"epoch: {0:>3} |\tstep: {0:>6} |\tloss: {' '*9}nan |\tSTSB: {best_stsb:.4f}") # 훈련 전이라 loss는 nan
    # 평가 결과 log를 저장
    logs: List[Dict[str, Union[int, float]]] = [
        {
            "epoch": 0,
            "step": best_step,
            "loss": None,
            "stsb": best_stsb,
        }
    ]

    # finally, start training!
    for epoch in range(args.epochs):
        model.train()

        # tqdm makes it easy to visualize how well the training is progressing
        for step, batch in tqdm(
            enumerate(train_dataloader), # 배치를 step, batch로 반환
            total=len(train_dataloader), # tqdm에 배치의 수 알려줌(dataset size / batch size)
            dynamic_ncols=True, # 터미널 크기에 맞춰서 tqdm 진도 바 너비 자동 조절
        ):
            # transfer batch to the device (GPU로 배치 이동)
            batch: BatchEncoding = batch.to(args.device)
            # if you want to see the actual data, please uncomment the following line.
            # print(batch)
            # and also, if you want to see the actual input strings, please uncomment the following line.
            # print(tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True))

            ####################################################################################################### 1. simply forward inputs twice!
            ####################################################################################################### different dropout masks are adapt automatically
            emb1 = model.forward(**batch)
            emb2 = model.forward(**batch)

            # SimCSE training objective:
            #    maximize the similarity between the same sentence
            # => make diagonal elements most similar

            ###################################################################################################### 2. similarity matrix 만들기
            # shape of sim_matrix: (batch_size, batch_size) 
            # calculate cosine similarity between all pair of embeddings (n x n)
            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
            '''
            1. unsqueeze 차원 확장
            emb1.unsqueeze(1): emb1의 차원을 (batch_size, 1, hidden_size)로 확장합니다. (for문 한번 당 배치 하나)
            emb2.unsqueeze(0): emb2의 차원을 (1, batch_size, hidden_size)로 확장합니다.
            
            2. broadcasting (batch_size, batch_size, hidden_size)

            3. dim=-1: 마지막 차원에 대해 코사인 유사도를 계산합니다.
            결과적으로 sim_matrix의 크기는 (batch_size, batch_size)가 됩니다.
            >> selbst_text.ipynb 참고
            '''
            # FYI: SimCSE is sensitive for the temperature parameter.
            '''
            비지도 대조 학습에는 uniformity-tolerance균일성-내성 딜레마가 존재하며, 
            temperature은 임베딩 분포의 local separation국부적 분리와 global uniformity전역적 균일성을 제어하는 데 핵심적인 역할을 합니다. 
            따라서 온도 선택은 모델이 분리 가능한 특징 사이의 균형을 맞추고 의미적으로 유사한 샘플에 대해 내성을 갖도록 만드는 데 중요합니다.
            So the choice of temperature is important to make model be balance between separable features and tolerant to semantically similar samples.

            <Understanding the Behaviour of Contrastive Loss>
            The temperature plays a role in controlling the strength of penalties on the hard negative samples. 
            Specifically, contrastive loss with small temperature tends to penalize much more on the hardest negative samples 
            such that the local structure of each sample tends to be more separated, and the embedding distribution is likely to be more
            uniform. On the other hand, contrastive loss with large temperature is less sensitive to the hard negative samples, and
            the hardness-aware property disappears as the temperature approaches +∞.

            <굳이 similarity에 나누는 이유?>
            𝜏가 작을수록 분포는 더 뾰족해져서 모델이 가장 유사한 (양의) 샘플과 가장 비유사한 (음의) 샘플에 집중하게 됩니다. 반대로, 𝜏가 클수록 분포는 더 평탄해집니다.
            기울기 계산시, 𝜏가 작을수록 어려운 음의 샘플에 대한 그라디언트가 커져서 더 강한 패널티를 주게 됩니다.
            엔트로피 관점에서, 낮은 온도는 엔트로피를 줄여 모델을 더 결정적으로 만들고, 높은 온도는 엔트로피를 증가시켜 모델을 더 불확실하게 하고 탐색적으로 만듭니다.
            '''
            ###################################################################################################### 3. similarity matrix에 temperature scaling
            # see Table D.1 of the paper (0.05가 가장 좋았다고 함)
            sim_matrix = sim_matrix / args.temperature

            ###################################################################################################### 4. label 값 만들기
            # sim_matrix 크기 (batch_size, batch_size)이니까 label도 (batch_size)만큼 있어야함
            '''
            대조 학습의 개념
            양성 예제(Positive Example): 동일한 문장의 서로 다른 표현(예: 같은 문장을 두 번 전처리하여 얻은 두 개의 임베딩).
            음성 예제(Negative Example): 다른 문장들로부터 얻은 임베딩.

            배치 내의 모든 문장 쌍에 대해 코사인 유사도를 계산한 행렬 sim_matrix의 크기는 (batch_size, batch_size)입니다. 
            행렬의 각 요소 sim_matrix[i, j]는 emb1[i]와 emb2[j]의 코사인 유사도를 나타냅니다.
            대각선 요소 sim_matrix[i, i]는 같은 문장의 두 임베딩(즉, 양성 예제)의 유사도를 나타냅니다.
            
            배치 크기가 4인 경우를 가정해 보겠습니다:
            - `sim_matrix`는 다음과 같은 형태를 가집니다:
  
                \[
                \begin{bmatrix}
                \text{sim}(emb1_0, emb2_0) & \text{sim}(emb1_0, emb2_1) & \text{sim}(emb1_0, emb2_2) & \text{sim}(emb1_0, emb2_3) \\
                \text{sim}(emb1_1, emb2_0) & \text{sim}(emb1_1, emb2_1) & \text{sim}(emb1_1, emb2_2) & \text{sim}(emb1_1, emb2_3) \\
                \text{sim}(emb1_2, emb2_0) & \text{sim}(emb1_2, emb2_1) & \text{sim}(emb1_2, emb2_2) & \text{sim}(emb1_2, emb2_3) \\
                \text{sim}(emb1_3, emb2_0) & \text{sim}(emb1_3, emb2_1) & \text{sim}(emb1_3, emb2_2) & \text{sim}(emb1_3, emb2_3)
                \end{bmatrix}
                \]

            - `labels`는 `[0, 1, 2, 3]` 형태를 가집니다.

            크로스 엔트로피 손실 함수는 `sim_matrix`의 각 행에서 `labels`에 해당하는 인덱스의 값을 최대화하도록 학습합니다. 즉, 각 행의 대각선 요소가 최대화되어 같은 문장의 두 임베딩이 가장 유사하도록 학습됩니다.
            '''
            # labels := [0, 1, 2, ..., batch_size - 1]
            # labels indicate the index of the diagonal element (i.e. positive examples)
            labels = torch.arange(args.batch_size).long().to(args.device)

            ###################################################################################################### 5. cross-entropy loss(softmax+max_sim)
            # it may seem strange to use Cross-Entropy Loss here.
            # this is a shorthund of doing SoftMax and maximizing the similarity of diagonal elements
            # 소프트맥스를 적용하고 diagonal elements의 similarity를 최대화하는 것을 간략화한 방식
            loss = F.cross_entropy(sim_matrix, labels)
            
            ###################################################################################################### 6. optimizer.. training
            optimizer.zero_grad() # 초기화해서 이전 배치 gradient 누적되지 않도록
            loss.backward() # gradient 계산
            optimizer.step() # 파라미터 update
            lr_scheduler.step() # lr 조정

            ###################################################################################################### 7. evaluation per eval_logging_interval
            # for every `args.eval_logging_interval` steps, perform evaluation on STS task and print logs
            # 마지막 배치 끝날때도 평가
            if (step + 1) % args.eval_logging_interval == 0 or (step + 1) == len(train_dataloader):
                model.eval() # evlauation 모드로 전환해야함!!
                # evaluate on the STS-B development set
                stsb_score = sts.dev(encode=encode)

                # you should use the best model for the evaluation to avoid using overfitted model
                # FYI: https://github.com/princeton-nlp/SimCSE/issues/62
                # 새로운게 이 전에 best점수보다 높으면 best점수 갱신
                if best_stsb < stsb_score:
                    best_stsb = stsb_score
                    best_step = step + 1 # 어느 step에 best점수 갱신됐는지
                    # only save the best performing model
                    torch.save(model.state_dict(), args.output_dir / "model.pt")

                # use `tqdm.write` instead of `print` to prevent terminal display corruption
                tqdm.write(
                    f"epoch: {epoch:>3} |\tstep: {step+1:>6} |\tloss: {loss.item():.10f} |\tSTSB: {stsb_score:.4f}"
                )
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss.item(),
                        "stsb": stsb_score,
                    }
                )
                pd.DataFrame(logs).to_csv(args.output_dir / "logs.csv", index=False)

                # if you want to see the changes of similarity matrix, uncomment the following line
                # tqdm.write(str(sim_matrix))
                ###################################################################################################### 8. 다시 또 training할 준비
                model.train() # 훈련 모드로 전환 (다음 배치에 대해 훈련 계속할 수 있게)

    # save epochs, steps, losses, and STSB dev scores
    with (args.output_dir / "dev-metrics.json").open("w") as f:
        data = {
            "best-step": best_step,
            "best-stsb": best_stsb,
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

    # load the best model for final evaluation
    if (args.output_dir / "model.pt").exists():
        model.load_state_dict(torch.load(args.output_dir / "model.pt"))
    model.eval().to(args.device)

    sts_metrics = sts(encode=encode)
    with (args.output_dir / "sts-metrics.json").open("w") as f:
        json.dump(sts_metrics, f, indent=2, ensure_ascii=False)

    with (args.output_dir / "config.json").open("w") as f:
        data = {k: v if type(v) in [int, float] else str(v) for k, v in vars(args).items()}
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
