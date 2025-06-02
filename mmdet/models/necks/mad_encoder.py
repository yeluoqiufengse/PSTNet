import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块


class MAD(nn.Module):  
    def __init__(self, channels=32,dilation=2, c2=None, factor=8):  # 构造函数，初始化对象
        super(MAD, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        #使用扩展卷积代替原始3×3卷积
        self.conv_dilated=nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=dilation,dilation=dilation)
        #新增全局上下文卷积分支
        self.global_context_conv=nn.Conv2d(channels, channels, kernel_size=1, stride=1,
                                 padding=0)

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        #1.生成全局上下文特征
        global_context=self.global_context_conv(self.agp(x)) #b,c,1,1
        #2.分组特征计算
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        #使用1×1卷积进行基础特征提取
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        # print(hw.shape)# 加空洞是torch.Size([32, 1, 56, 1])，不加是torch.Size(torch.Size([32, 1, 56, 1]))
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        #基于1×1和sigmod加权的特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        #使用扩展卷积代替原始3×3卷积
        x2=self.conv_dilated(group_x)
        # x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        # print(x2.shape)#加空洞是torch.Size([32, 1, 26, 26])，不加是torch.Size([32, 1, 28, 28])
        #生成加权注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        #结合全局上下文特征进行加权融合
        final_output=(group_x * weights.sigmoid()).reshape(b,c,h,w)
        return final_output+global_context.expand_as(final_output)  #加入全局上下文特征

if __name__ == '__main__':
    MAD=MAD()
    x=torch.randn(1,32,28,28)
    size=MAD(x).shape
    print(size)
