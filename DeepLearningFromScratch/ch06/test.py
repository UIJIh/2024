import matplotlib.pyplot as plt
import numpy as np

pi=3.14
xx=np.linspace(0,2*pi,100)


fig = plt.figure()
ax  = fig.add_subplot(111)

phases=[0,pi/4,pi/2]
markers=['s','o','^']

for phase,mk in zip(phases,markers):
    labeltext='Sine' + '*' + str(phase)
    F=[np.sin(x+phase) for x in xx]
    ax.plot(xx,F,color='b',marker=mk,label=labeltext)

    labeltext='Cosine' + '*' + str(phase)
    F=[np.cos(x+phase) for x in xx]
    ax.plot(xx,F,color='g',marker=mk,label=labeltext)

hand, labl = ax.get_legend_handles_labels()
#hand, labl = function_to_split(hand,labl,'*')
ax.legend(hand,labl)
plt.show()