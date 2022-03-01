import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
plt.style.use('seaborn-white')

x_major_locator=MultipleLocator(0.05)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(2)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)

plt.grid()
data = [9,12,5,15,5,0,2,1,3,1,2,2,2,0,1,1,1,0,5,4,0]
bins = [round(0.05 * i, 2) for i in range(21)]
rects1 = plt.bar(x=bins , height=data, width=0.05, alpha=1.0, edgecolor='k', color=np.array([142, 183, 213])/255)
plt.ylim(0, 16)     # y轴取值范围
plt.ylabel("Total Number")
plt.xlabel("Loss Value")

x = [bins[i] for i in range(len(bins)) if i % 4 == 0]
plt.xticks([index - 0.025 for index in x], x)
plt.tick_params(labelsize=12)
plt.savefig('data.pdf', bbox_inches='tight', pad_inches = 1)
plt.show()
