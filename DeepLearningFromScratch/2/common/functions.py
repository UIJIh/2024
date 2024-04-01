# coding: utf-8
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    '''
    - 2차원 입력의 경우:
    먼저, 각 행에서 최대값을 빼줍니다(x - x.max(axis=1, keepdims=True)). 이는 수치적으로 안정된 소프트맥스 계산을 위한 기법으로, 오버플로우를 방지합니다.
    그 다음, np.exp(x)를 사용하여 각 요소의 지수 값을 계산합니다.
    마지막으로, 각 행의 합으로 각 요소를 나누어 정규화합니다(x /= x.sum(axis=1, keepdims=True)).

    - 1차원 입력의 경우:
    최대값을 빼줍니다(x - np.max(x)). 이는 2차원 입력의 경우와 마찬가지로 수치적 안정성을 위함입니다.
    np.exp(x)를 사용하여 지수 값을 계산하고, 이를 전체 합으로 나누어 정규화합니다(np.exp(x) / np.sum(np.exp(x))).
    '''
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
