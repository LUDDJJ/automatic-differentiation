import math
from graphviz import Digraph

# ==========================================
# 1. 你的类定义 (稍作增强以支持可视化)
# ==========================================

class Value_forward:
    def __init__(self, data, grad=0.0, label=''):
        self.data = data
        self.grad = grad
        self.label = label # 增加 label 方便看图

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value_forward) else Value_forward(other, 0.0)
        # 记录运算过程用于 label
        out = Value_forward(self.data + other.data, self.grad + other.grad, label=f"({self.label}+{other.label})")
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value_forward) else Value_forward(other, 0.0)
        out = Value_forward(self.data * other.data, self.data * other.grad + other.data * self.grad, label=f"({self.label}*{other.label})")
        return out
    
    def sin(self):
        out = Value_forward(math.sin(self.data), math.cos(self.data) * self.grad, label=f"sin({self.label})")
        return out

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward
        return out
    
    def pow(self, exponent):
        out = Value(self.data ** exponent, (self,), f'**{exponent}')
        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# ==========================================
# 2. 可视化函数：反向传播 (Reverse Mode)
# ==========================================
def draw_reverse_graph(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'}) # LR = Left to Right
    
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    
    for n in nodes:
        # 节点显示：数值 | 梯度
        uid = str(id(n))
        # 红色代表梯度，蓝色代表数据
        label = f"{{ {n.label} | data {n.data:.2f} | <grad> grad {n.grad:.2f} }}"
        dot.node(name=uid, label=label, shape='record', style='filled', fillcolor='white')
        
        if n._op:
            # 创建一个操作符小圆圈
            op_uid = uid + n._op
            dot.node(name=op_uid, label=n._op, shape='circle', style='filled', fillcolor='orange')
            dot.edge(op_uid, uid)
            
    for n1, n2 in edges:
        # 连接线
        n1_uid = str(id(n1))
        n2_uid = str(id(n2))
        op_uid = n2_uid + n2._op
        dot.edge(n1_uid, op_uid)
        
    return dot

# ==========================================
# 3. 可视化模拟：前向微分 (Forward Mode)
# ==========================================
# 前向微分没有图，所以我们用表格+步骤图来表示
def visualize_forward_steps():
    print("="*60)
    print("前向微分 (Forward Mode) 逐级追踪")
    print("计算公式: L = (x * y + sin(x))^2")
    print("求导目标: dL/dx (所以 x.grad 初始化为 1.0)")
    print("="*60)
    
    # 步骤记录
    steps = []
    
    # Step 1: 初始化
    x = Value_forward(2.0, 1.0, "x")
    y = Value_forward(5.0, 0.0, "y")
    steps.append(("Input", x, y, "x=2, x'=1 (种子); y=5, y'=0 (常数)"))
    
    # Step 2: 乘法
    w1 = x * y
    w1.label = "x*y"
    steps.append(("Mult", w1, None, f"Data: 2*5={w1.data:.2f} | Grad: 1*5 + 2*0 = {w1.grad:.2f}"))
    
    # Step 3: Sin
    w2 = x.sin()
    w2.label = "sin(x)"
    steps.append(("Sin", w2, None, f"Data: sin(2)={w2.data:.2f} | Grad: cos(2)*1 = {w2.grad:.2f}"))
    
    # Step 4: Add
    w3 = w1 + w2
    w3.label = "w1+w2"
    steps.append(("Add", w3, None, f"Data: 10+0.91={w3.data:.2f} | Grad: 5+(-0.42) = {w3.grad:.2f}"))
    
    # Step 5: Pow (模拟 pow 运算)
    # 你的类暂时没写 pow，我手动模拟一下 __mul__ self
    # L = w3 * w3 
    L = Value_forward(w3.data**2, 2*w3.data*w3.grad, label="L") 
    steps.append(("Pow", L, None, f"Data: 10.91^2={L.data:.2f} | Grad: 2*10.91*4.58 = {L.grad:.2f}"))

    # 打印表格
    print(f"{'Step':<10} | {'Node':<10} | {'Value (Data)':<15} | {'Derivative (Grad)':<20}")
    print("-" * 65)
    for name, node, _, desc in steps:
        if node:
            print(f"{name:<10} | {node.label:<10} | {node.data:<15.4f} | {node.grad:<20.4f}")
            
    return steps

# ==========================================
# 4. 执行并生成
# ==========================================

# A. 生成反向传播图
x = Value(2.0, label='x')
y = Value(5.0, label='y')
L = (x * y + x.sin()).pow(2)
L.label = 'L'
L.backward()

try:
    dot = draw_reverse_graph(L)
    dot.render('reverse_mode_graph', view=True, cleanup=True)
    print("\n[INFO] 反向传播图已生成: reverse_mode_graph.png")
except Exception as e:
    print(f"\n[ERROR] Graphviz 未安装或路径未配置，无法生成图片: {e}")

# B. 打印前向传播过程
visualize_forward_steps()