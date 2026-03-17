import math
import os
import numpy as np
import scipy
import scipy.optimize
from air_track.aligner.utils.transform_utils import build_geom_transform, create_points
from sklearn.linear_model import LinearRegression
from air_track.utils import common_utils, combine_load_cfg_yaml, reprod_init


def offset_grid_to_transform_params(prev_frame_points, cur_frame_points, points_weight):
    """
    :param prev_frame_points: ndarray, shape 2xN, coordinates of points on the prev frame
    :param cur_frame_points: ndarray, shape 2xN, coordinates of points on the current frame
    :param points_weight: ndarray, shape N, weight of each point
    :return: dx, dy, angle, so
    cur_frame_points = T(dx,dy,angle) * prev_frame_points

    points have zero at the image center
    """
    points_weight = points_weight.astype(np.double) / points_weight.sum()

    dxy0 = -1 * ((cur_frame_points - prev_frame_points) * points_weight[None, :]).sum(axis=1)

    cx = cur_frame_points[0].astype(np.double)
    cy = cur_frame_points[1].astype(np.double)

    px = prev_frame_points[0].astype(np.double)
    py = prev_frame_points[1].astype(np.double)

    def cost(x):
        """成本函数"""
        dx, dy, a = x
        a = a / 1000.0

        pred_px = cx - a * cy + dx
        pred_py = a * cx + cy + dy

        # 加权欧氏距离的平方
        err = points_weight * ((px - pred_px) ** 2 + (py - pred_py) ** 2)
        return err.sum()

    def der(x):
        """导数/梯度函数"""
        dx, dy, a = x
        a = a / 1000.0

        pred_px = cx - a * cy + dx
        pred_py = a * cx + cy + dy

        # 求导
        ddx = points_weight * 2 * (pred_px - px)
        ddy = points_weight * 2 * (pred_py - py)
        dda = points_weight * 2 * (cx * (pred_py - py) - cy * (pred_px - px)) / 1000.0
        return ddx.sum(), ddy.sum(), dda.sum()

    x0 = np.array([dxy0[0], dxy0[1], 0.0])

    # 最小化成本函数cost的参数x，即找到的使成本函数最小的参数向量
    # options={'gtol': 1e-6, 'disp': False}：gtol为优化过程的收敛条件，disp为是否显示优化过程的信息
    res = scipy.optimize.minimize(cost, x0, jac=der, method='BFGS', options={'gtol': 1e-6, 'disp': False})

    # 返回优化的参数值x[0]、x[1]、x[2]、目标函数的优化值，即成本函数在最优参数处的值
    return res.x[0], res.x[1], res.x[2] / 1000.0 * 180 / math.pi, res.fun  # cost(res.x)


def offset_grid_to_transform(prev_frame_points, cur_frame_points, points_weight):
    """
    :param prev_frame_points: ndarray, shape 2xN, coordinates of points on the prev frame
    :param cur_frame_points: ndarray, shape 2xN, coordinates of points on the current frame
    :param points_weight: ndarray, shape N, weight of each point
    :return: T, so
    cur_frame_points = T * [prev_frame_points.T, 1].T
    """
    points_weight = points_weight.astype(np.double) / points_weight.sum()

    # 线性回归
    mx = LinearRegression()
    mx.fit(prev_frame_points.transpose(), cur_frame_points[0], sample_weight=points_weight)
    my = LinearRegression()
    my.fit(prev_frame_points.transpose(), cur_frame_points[1], sample_weight=points_weight)

    t = np.array([
        [mx.coef_[0], mx.coef_[1], mx.intercept_],
        [my.coef_[0], my.coef_[1], my.intercept_],
        [0, 0, 1]
    ])
    return t, 0


def test_offset_grid_to_transform_params(point_shape, img_w, img_h):
    """
    测试offset_grid_to_transform_params函数
    这种方法误差较大，且很容易由于随机数种子的不一样导致误差不满足要求
    """
    prev_points, prev_points_1d = create_points(point_shape, crop_w=img_w, crop_h=img_h)

    dx = 64
    dy = 42
    angle = 0.5

    transform = build_geom_transform(translation_x=dx,
                                     translation_y=dy,
                                     scale_x=1.0,
                                     scale_y=1.0,
                                     angle=angle,
                                     return_params=True)

    cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T
    # cur_points = cur_points.reshape((2, 32, 32))

    points_weight = np.ones((prev_points_1d.shape[1],), dtype=np.float32)

    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d, cur_points, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.01
    assert abs(dy_pred - dy) < 0.01
    assert abs(angle_pred - angle) < 0.001
    assert err < 0.001

    cur_points = np.random.normal(cur_points, np.ones_like(cur_points))

    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d, cur_points, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.1
    assert abs(dy_pred - dy) < 0.1
    assert abs(angle_pred - angle) < 0.01
    expected_err = 2.0
    assert abs(err - expected_err) < 0.25

    # test with the custom weight
    points_weight[:len(points_weight) // 2] = 2.0
    with common_utils.timeit_context('estimate transformation'):
        dx_pred, dy_pred, angle_pred, err = offset_grid_to_transform_params(prev_points_1d, cur_points, points_weight)

    print(dx_pred, dy_pred, angle_pred, err)
    assert abs(dx_pred - dx) < 0.1
    assert abs(dy_pred - dy) < 0.1
    assert abs(angle_pred - angle) < 0.01


def test_offset_grid_to_transform(point_shape, img_w, img_h):
    """测试offset_grid_to_transform函数"""
    prev_points, prev_points_1d = create_points(point_shape, crop_w=img_w, crop_h=img_h)

    dx = 64
    dy = 42
    angle = 0.5

    transform = build_geom_transform(translation_x=dx,
                                     translation_y=dy,
                                     scale_x=1.0,
                                     scale_y=1.0,
                                     angle=angle,
                                     return_params=True)

    cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T
    # cur_points = cur_points.reshape((2, 32, 32))

    points_weight = np.ones((prev_points_1d.shape[1],), dtype=np.float32)

    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T

    print(transform)
    print(t)
    print(err, np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points).max())
    assert np.abs(t - transform).max() < 1e-3
    assert np.abs(cur_points_pred - cur_points).max() < 0.1

    print('Add noise')
    cur_points_orig = cur_points
    cur_points = np.random.normal(cur_points, np.ones_like(cur_points))

    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    print(t)
    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T
    print(np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points_orig).max())
    assert np.abs(t - transform).max() < 0.2
    assert np.abs(cur_points_pred - cur_points_orig).max() < 1

    # test with the custom weight
    points_weight[:len(points_weight) // 2] = 2.0
    with common_utils.timeit_context('estimate transformation'):
        t, err = offset_grid_to_transform(prev_points_1d, cur_points, points_weight)

    print(t)
    cur_points_pred = ((t[:2, :2] @ prev_points_1d).T + t[:2, 2]).T
    print(np.abs(t - transform).max())
    print(np.abs(cur_points_pred - cur_points_orig).max())
    assert np.abs(t - transform).max() < 0.2
    assert np.abs(cur_points_pred - cur_points_orig).max() < 1


def test_transform_params_and_transform(point_shape, img_w, img_h):
    """
    使用模型真实输出，来测试对比上边两种方法的误差
    offsets和heatmaps为模型输出真实数据
    """
    prev_points, prev_points_1d = create_points(point_shape, crop_w=img_w, crop_h=img_h)
    prev_points = prev_points[:, 2: -2, 2: -2]

    offsets = np.array([[[1.14324927e+00, 6.06066883e-01, 4.97711241e-01,
                          5.88206470e-01, 7.30900466e-01, 5.44514537e-01,
                          4.30881679e-01, 2.86635071e-01, 3.67837578e-01,
                          5.96428752e-01, 6.68007851e-01, 5.87134421e-01,
                          6.29459918e-01, 6.32953763e-01, 5.18279850e-01,
                          2.69746840e-01],
                         [7.61049092e-01, 6.03332341e-01, 6.34099662e-01,
                          6.20802283e-01, 6.84092581e-01, 5.65024137e-01,
                          3.87991846e-01, 2.62652993e-01, 4.35171664e-01,
                          5.49276412e-01, 6.23876750e-01, 4.36610520e-01,
                          5.84040821e-01, 5.36355555e-01, 3.60153764e-01,
                          -3.70309837e-02],
                         [8.46725166e-01, 6.86369181e-01, 6.83110118e-01,
                          7.11788476e-01, 8.52485836e-01, 6.65595651e-01,
                          5.18963039e-01, 5.12819648e-01, 7.10620701e-01,
                          7.79424667e-01, 1.02463198e+00, 8.77224207e-01,
                          1.24507868e+00, 9.90526736e-01, 6.77451313e-01,
                          3.65575492e-01],
                         [4.63947952e-01, 5.09369731e-01, 5.56311429e-01,
                          5.30459166e-01, 5.60256541e-01, 3.80022526e-01,
                          3.62100363e-01, 5.16803265e-01, 6.46166861e-01,
                          6.64396942e-01, 8.57338309e-01, 5.06490886e-01,
                          8.29675853e-01, 5.51386952e-01, 2.97388941e-01,
                          1.53720126e-01],
                         [7.12680280e-01, 7.75783360e-01, 6.34830415e-01,
                          6.16348207e-01, 5.79109192e-01, 4.79137659e-01,
                          4.41579580e-01, 6.13301098e-01, 6.80567086e-01,
                          6.48550093e-01, 8.15539479e-01, 4.73908424e-01,
                          8.53743434e-01, 6.99343204e-01, 2.75858283e-01,
                          1.04156092e-01],
                         [5.95164061e-01, 3.70529294e-01, 3.16638947e-01,
                          4.55963850e-01, 5.09198129e-01, 2.91364908e-01,
                          3.10754597e-01, 4.29327846e-01, 3.38701844e-01,
                          3.17612827e-01, 5.09129167e-01, 2.10824504e-01,
                          6.99964523e-01, 6.03954911e-01, 1.26209691e-01,
                          3.23615372e-01],
                         [6.96655333e-01, 4.97781932e-01, 5.89177668e-01,
                          6.38915837e-01, 6.83578908e-01, 3.23244631e-01,
                          4.28235650e-01, 5.45853376e-01, 4.72231090e-01,
                          4.22547877e-01, 4.96671140e-01, 1.94101289e-01,
                          6.34363413e-01, 5.59309185e-01, 1.83551744e-01,
                          5.31636834e-01],
                         [5.95262408e-01, 6.41699672e-01, 7.32091844e-01,
                          7.71638751e-01, 9.65239108e-01, 6.84257984e-01,
                          7.94148386e-01, 8.48186672e-01, 4.70426440e-01,
                          3.98206174e-01, 3.67223024e-01, 1.64662138e-01,
                          8.08432579e-01, 7.74823189e-01, 4.31494355e-01,
                          6.86163545e-01],
                         [7.02013671e-01, 7.90161312e-01, 7.95141339e-01,
                          9.04415011e-01, 1.06579089e+00, 8.21720421e-01,
                          8.30979824e-01, 7.83825099e-01, 4.59318399e-01,
                          3.91357064e-01, 5.91199696e-01, 4.29370463e-01,
                          8.60853970e-01, 7.09890008e-01, 3.06422383e-01,
                          5.21166146e-01],
                         [6.16213858e-01, 6.46323442e-01, 7.18704581e-01,
                          7.50589490e-01, 8.41875792e-01, 5.29462874e-01,
                          6.71100080e-01, 6.25986814e-01, 5.31456292e-01,
                          4.87442017e-01, 5.23363888e-01, 4.10944521e-01,
                          7.10328698e-01, 4.66536164e-01, -2.38592610e-01,
                          -2.56394327e-01],
                         [6.71179533e-01, 5.21940172e-01, 5.23618340e-01,
                          6.29071295e-01, 5.11069715e-01, 1.48692861e-01,
                          3.59271616e-01, 3.46769094e-01, 3.46316457e-01,
                          4.79220033e-01, 4.52466249e-01, 3.06463055e-02,
                          3.55201095e-01, 4.69157934e-01, -3.38524371e-01,
                          -4.01756823e-01],
                         [8.83906782e-01, 5.79150140e-01, 4.98632014e-01,
                          5.43324769e-01, 4.52797294e-01, 2.65378118e-01,
                          4.88796502e-01, 5.20327628e-01, 2.98911929e-01,
                          3.00185680e-01, -2.06947513e-02, -1.45965561e-01,
                          4.40120697e-01, 6.65432751e-01, -8.24416429e-02,
                          -2.84059018e-01]],
                        [[8.05879951e-01, 5.30230582e-01, 2.74886400e-01,
                          1.86749145e-01, 1.86864302e-01, 8.66147056e-02,
                          5.92047684e-02, -2.56095417e-02, -6.28309371e-03,
                          1.82846591e-01, 3.99588197e-01, 2.69386977e-01,
                          1.13042928e-01, 1.42425999e-01, 1.98517367e-01,
                          -1.21188216e-01],
                         [5.58244109e-01, 4.90742534e-01, 9.74314287e-02,
                          5.68444245e-02, -7.55036101e-02, -2.06324771e-01,
                          -2.55865753e-01, -3.30764085e-01, -2.43677273e-01,
                          -6.51967712e-05, 1.07870795e-01, -4.35058363e-02,
                          -2.77784944e-01, -1.86474696e-01, 1.53024076e-03,
                          -1.51747093e-01],
                         [7.13500753e-02, 9.97534767e-02, -1.00214146e-01,
                          -1.65510193e-01, -2.91609079e-01, -3.98848802e-01,
                          -3.91250521e-01, -4.60796267e-01, -3.22789818e-01,
                          -1.36713639e-01, -8.31265468e-04, -2.83629209e-01,
                          -6.10452175e-01, -6.02770805e-01, -3.32175732e-01,
                          -6.73007369e-01],
                         [1.69706866e-01, 5.61424904e-02, -3.50646861e-02,
                          -1.84957221e-01, -2.91279316e-01, -2.53995895e-01,
                          -2.00552806e-01, -6.92836568e-02, 9.90520492e-02,
                          3.35713536e-01, 5.65621197e-01, 1.94610462e-01,
                          -2.22646430e-01, -4.11426812e-01, -2.67359823e-01,
                          -6.35638237e-01],
                         [3.64961833e-01, 8.70826617e-02, -5.32943197e-02,
                          -2.12114111e-01, -2.48497382e-01, -1.43952742e-01,
                          -2.39524052e-01, -2.04134002e-01, -1.41815618e-01,
                          -4.65269573e-02, 7.26871863e-02, -1.86026528e-01,
                          -2.94918984e-01, -3.52532059e-01, -1.98278978e-01,
                          -6.55484200e-01],
                         [1.75666198e-01, 3.25701050e-02, -1.75315395e-01,
                          -3.33586723e-01, -3.23231608e-01, -2.40880445e-01,
                          -4.01151091e-01, -5.33487856e-01, -6.32399261e-01,
                          -6.22337997e-01, -5.31906128e-01, -1.03085434e+00,
                          -1.07100821e+00, -8.70827615e-01, -5.60054302e-01,
                          -8.84814978e-01],
                         [-7.64167234e-02, -9.91276726e-02, -3.54207367e-01,
                          -4.34197217e-01, -5.84062099e-01, -5.00658989e-01,
                          -4.78536397e-01, -4.03098553e-01, -1.95518211e-01,
                          1.32748298e-02, -1.00506149e-01, -7.64902413e-01,
                          -9.73726451e-01, -1.08541667e+00, -1.02174747e+00,
                          -1.37129402e+00],
                         [-1.77720562e-01, -1.57738641e-01, -2.72327751e-01,
                          -5.57852507e-01, -7.91942179e-01, -8.16390753e-01,
                          -6.29653573e-01, -5.74375510e-01, -2.30927691e-01,
                          -8.49463567e-02, -3.07563871e-01, -5.49070477e-01,
                          -7.09278882e-01, -6.91046000e-01, -7.91667640e-01,
                          -1.20274305e+00],
                         [7.32248798e-02, -9.02027711e-02, -3.32535237e-01,
                          -6.60182416e-01, -9.85131860e-01, -9.90481317e-01,
                          -7.76212692e-01, -7.44753420e-01, -4.90260601e-01,
                          -4.28579718e-01, -5.56336105e-01, -6.10235870e-01,
                          -7.68715382e-01, -4.58846271e-01, -4.97945696e-01,
                          -1.23837340e+00],
                         [1.12481751e-01, 1.52683072e-02, -2.78519303e-01,
                          -6.08831644e-01, -9.18783903e-01, -9.89580274e-01,
                          -7.66334951e-01, -8.12159538e-01, -7.44883895e-01,
                          -6.40468121e-01, -4.47360277e-01, -5.17159283e-01,
                          -9.15783763e-01, -5.31741977e-01, -4.94112521e-01,
                          -1.19287622e+00],
                         [-2.41143890e-02, -2.04975948e-01, -5.38593352e-01,
                          -8.62675250e-01, -1.06489861e+00, -1.01216006e+00,
                          -8.49480093e-01, -8.93326104e-01, -9.12369311e-01,
                          -1.06936467e+00, -8.53434861e-01, -7.57276177e-01,
                          -9.62970018e-01, -4.66962188e-01, -4.03292716e-01,
                          -1.21284020e+00],
                         [2.98031956e-01, 3.86833362e-02, -6.47607267e-01,
                          -9.73123431e-01, -1.09994709e+00, -1.00111163e+00,
                          -7.98177540e-01, -6.86802447e-01, -8.12072754e-01,
                          -9.33492005e-01, -7.18323886e-01, -5.87624967e-01,
                          -7.24849343e-01, -4.51015204e-01, -5.12984216e-01,
                          -9.89296973e-01]]], dtype=np.float32)

    cur_points = prev_points + offsets

    heatmap = np.array([[[0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833],
                         [0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833, 0.00520833, 0.00520833, 0.00520833, 0.00520833,
                          0.00520833]]], dtype=np.float32)

    dx, dy, angle, err = offset_grid_to_transform_params(
        prev_frame_points=prev_points.reshape(2, -1),
        cur_frame_points=cur_points.reshape(2, -1),
        points_weight=heatmap.reshape(-1) ** 2
    )

    tr, _ = offset_grid_to_transform(
        prev_frame_points=prev_points.reshape(2, -1),
        cur_frame_points=cur_points.reshape(2, -1),
        points_weight=heatmap.reshape(-1) ** 2
    )

    transform = build_geom_transform(translation_x=dx, translation_y=dy,
                                     scale_x=1.0, scale_y=1.0,
                                     angle=angle,
                                     return_params=True)

    print('offset_grid_to_transform_params output: \n', transform)
    print('offset_grid_to_transform output: \n', tr)

    print(np.abs(tr[:2] - transform[:2]).max())


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_yaml = os.path.join(script_dir, 'config/dataset.yaml')
    train_yaml = os.path.join(script_dir, 'config/align_train.yaml')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    points_shape = eval(cfg_data['points_shape'])
    img_size_w, img_size_h = cfg_data['dataset_params']['img_size']

    # 固定随机数种子
    reprod_init(seed=cfg_data['seed'])

    # 运行三种测试方式
    test_offset_grid_to_transform_params(points_shape, img_w=img_size_w, img_h=img_size_h)
    test_offset_grid_to_transform(points_shape, img_w=img_size_w, img_h=img_size_h)
    test_transform_params_and_transform(points_shape, img_w=img_size_w, img_h=img_size_h)
