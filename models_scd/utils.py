import torch

def transform_view(x, n_columns):
    '''
    3d -> 2d , 2d -> 3d
    3d -> 2d:
        第一条锚点样本           (x1)
        第二条锚点样本           (x2)
        第三条锚点样本           (x3)
        第一条正样本             [x1+)
        第二条正样本             [x2+)
        第三条正样本             [x3+)
        第一条负样本            【x1-)
        第二条负样本            【x2-)
        第三条负样本            【x3-)
    2d -> 3d:
    [
        第一条锚点样本           (x1)
        第一条正样本             [x1+)
        第一条负样本            【x1-)
    ]
    [   
        第二条锚点样本           (x2)
        第二条正样本             [x2+)
        第二条负样本            【x2-)
    ]
    [
        第三条锚点样本           (x3)
        第三条正样本             [x3+)
        第三条负样本            【x3-)
    ]
    '''
    if len(x.shape) == 2:  # 2d -> 3d
        sep = x.shape[0] / n_columns
        assert sep.is_integer()
        sep = int(sep)
        return torch.cat( [ x[sep * i : sep * (i + 1) ] for i in range(n_columns) ], dim=1).reshape([sep, n_columns, -1])
    elif len(x.shape) == 3:
        return torch.cat([ x[:, i] for i in range(n_columns)],dim=0)
    else:
        raise ValueError


# def transform_view(x, n_columns):
#     '''
#     3d -> 2d , 2d -> 3d
#     '''
#     if len(x.shape) == 2:  # 2d -> 3d
#         sep = x.shape[0] / n_columns
#         assert sep.is_integer()
#         sep = int(sep)
#         return torch.cat( [ x[sep * i : sep * (i + 1) ] for i in range(n_columns) ], dim=1).reshape([sep, n_columns, -1])
#         # if n_columns == 3:
#         #     return torch.cat([x[0: sep], x[sep: sep * 2], x[sep * 2 : ]], dim=1).reshape([sep, n_columns, -1])
#         # elif n_columns == 2:
#         #     return torch.cat([x[0: sep], x[sep: ]], dim=1).reshape([sep, n_columns, -1])
#         # elif n_columns == 4:
#         #     return torch.cat([x[0: sep], x[sep: sep * 2], x[sep * 2 : sep * 3], x[sep * 3 : ]], dim=1).reshape([sep, n_columns, -1])
#         # else:
#         #     raise ValueError
#     elif len(x.shape) == 3:
#         return torch.cat([ x[:, i] for i in range(n_columns)],dim=0)
#         # if n_columns == 3:
#         #     return torch.cat([x[:, 0],x[:, 1],x[:, 2]],dim=0)
#         # elif n_columns == 2:
#         #     return torch.cat([x[:, 0],x[:, 1]],dim=0)
#         # elif n_columns == 4:
#         #     return torch.cat([x[:, 0],x[:, 1],x[:, 2],x[:, 3]],dim=0)
#         # else:
#         #     raise ValueError
#     else:
#         raise ValueError