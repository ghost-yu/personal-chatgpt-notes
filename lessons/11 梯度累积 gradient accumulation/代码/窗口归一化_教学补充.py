# -*- coding: utf-8 -*-
"""独立教学补充：真实PyTorch梯度，窗口按实际样本数归一化。

原Markdown循环仍在原notebook；这里的修正版不冒充课程原代码。
不下载权重，不测GPU显存，输入是CPU上的小型二维回归数据。
"""
from itertools import islice
import torch


def linear_fixture():
    """创建独立float64 Linear(1,1)，设置w=1,b=0；不依赖随机初始值。"""
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    # no_grad仅控制赋初值不建图；训练forward仍正常记录自动求导。
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    return model


def parameter_vector(model):
    """只读参数，detach后拼接为[w,b]；返回Tensor[2]，不保留训练图。"""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_weighted_windows(model, batches, steps=4, learning_rate=0.1):
    """按真实样本数归一化，支持短窗口和不等micro大小。

    model为Linear(1,1)；batches可迭代二元组，每项float64 CPU[X,1]。
    steps默认4，仅确定窗口最大批数；返回每次更新前的梯度与更新边界。
    参数和optimizer状态在step时改变，输入不被修改；没有AMP/DDP。
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion_sum = torch.nn.MSELoss(reduction='sum')
    iterator = iter(batches)
    seen = 0
    reports = []
    while True:
        # islice最多取steps个CPU批次，不提前计算forward或保留多份计算图。
        window = list(islice(iterator, steps))
        if not window:
            break
        total_samples = sum(labels.shape[0] for _, labels in window)
        optimizer.zero_grad(set_to_none=True)
        for inputs, labels in window:
            # 本例每样本只有一个回归目标，因此sum/样本数等于样本平均损失。
            predictions = model(inputs)
            loss = criterion_sum(predictions, labels) / total_samples
            loss.backward()  # 加到参数.grad；不设置retain_graph。
            seen += 1
        gradient = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])
        reports.append({'after_micro_batch': seen, 'samples': total_samples,
                        'gradient': gradient.tolist()})
        optimizer.step()
    return reports


def run_checks(original_loops):
    """实际执行两个原Markdown循环与修正版，返回普通dict作为验证证据。"""
    # original_loops为从原Markdown5/7提取的源码字符串，完整执行不改循环体。
    device = torch.device('cpu')
    features = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float64)
    labels = torch.zeros_like(features)
    # 一次batch4获得基准梯度[15,5]，不使用手写导数代替autograd。
    reference = linear_fixture()
    torch.nn.MSELoss()(reference(features), labels).backward()
    big_gradient = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
    assert torch.equal(big_gradient, torch.tensor([15., 5.], dtype=torch.float64))

    loop_results = []
    for count, name in ((4, 'full_window'), (2, 'tail_window')):
        model = linear_fixture()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        data_loader = [(features[i:i+1], labels[i:i+1]) for i in range(count)]
        context = dict(torch=torch, model=model, optimizer=optimizer, criterion=criterion,
                       data_loader=data_loader, device=device)
        exec(compile(original_loops[7], '<original-Markdown7>', 'exec'), context)
        old_parameters = parameter_vector(model)
        corrected = linear_fixture()
        reports = train_weighted_windows(corrected, data_loader)
        correct_parameters = parameter_vector(corrected)
        if count == 4:
            assert torch.allclose(old_parameters, correct_parameters, atol=1e-14, rtol=0)
            assert torch.allclose(old_parameters, torch.tensor([-.5, -.5], dtype=torch.float64))
        else:
            # 原2批仍除4：SGD参数[.75,-.15]；实际平均应为[.5,-.3]。
            assert torch.allclose(old_parameters, torch.tensor([.75, -.15], dtype=torch.float64))
            assert torch.allclose(correct_parameters, torch.tensor([.5, -.3], dtype=torch.float64))
        loop_results.append({'case': name, 'original_parameters': old_parameters.tolist(),
                             'corrected_parameters': correct_parameters.tolist(), 'corrected_windows': reports})

    # 原普通loop每micro更新，在不同参数点求导，结果与batch4不同。
    vanilla = linear_fixture()
    context = dict(torch=torch, model=vanilla, optimizer=torch.optim.SGD(vanilla.parameters(), lr=.1),
                   criterion=torch.nn.MSELoss(), data_loader=[(features[i:i+1], labels[i:i+1]) for i in range(4)], device=device)
    exec(compile(original_loops[5], '<original-Markdown5>', 'exec'), context)
    assert torch.allclose(parameter_vector(vanilla), torch.tensor([.2112, -.4992], dtype=torch.float64))

    # 不等大小2+1：sum/总样本数正好匹配合并3个样本的真实梯度。
    unequal_x = torch.tensor([[1.], [2.], [10.]], dtype=torch.float64)
    unequal_y = torch.zeros_like(unequal_x)
    unequal = linear_fixture()
    reports = train_weighted_windows(unequal, [(unequal_x[:2], unequal_y[:2]), (unequal_x[2:], unequal_y[2:])])
    expected = linear_fixture()
    torch.nn.MSELoss()(expected(unequal_x), unequal_y).backward()
    grad = torch.cat([p.grad.reshape(-1) for p in expected.parameters()])
    assert torch.allclose(torch.tensor(reports[0]['gradient'], dtype=torch.float64), grad, atol=1e-13, rtol=0)

    # 10个micro真实执行，窗口边界4/8/10，每次forward后立即backward。
    ten = linear_fixture()
    ten_windows = train_weighted_windows(ten, [(features[:1], labels[:1])] * 10)
    assert [r['after_micro_batch'] for r in ten_windows] == [4, 8, 10]

    # 使用同一图第二次反向需retain_graph；梯度清零与新图则不需。
    x = torch.tensor(1., requires_grad=True)
    y = x ** 2
    y.backward(retain_graph=True)
    first = x.grad.item()
    y.backward()
    second = x.grad.item()
    try:
        y.backward()
    except RuntimeError as error:
        freed_graph_error = str(error)
    else:
        raise AssertionError('Expected freed graph error')
    x.grad = None
    (x ** 2).backward()
    assert x.grad.item() == 2 and first == 2 and second == 4
    return {'scope': 'actual CPU float64 Linear/SGD/MSE and original Markdown loops; no BERT-large benchmark',
            'big_gradient': big_gradient.tolist(), 'original_and_corrected': loop_results,
            'vanilla_each_micro_parameters': parameter_vector(vanilla).tolist(),
            'unequal_window_gradient': reports[0]['gradient'], 'ten_windows': ten_windows,
            'same_graph_grads': [first, second], 'freed_graph_error': freed_graph_error,
            'new_graph_after_grad_None': x.grad.item()}
