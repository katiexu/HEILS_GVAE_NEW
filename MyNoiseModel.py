import copy

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError
)

TWO_QUBIT_GATES = {'cx', 'cz', 'rzz', 'ecr'}

# ==============================================
# 【芯片噪声配置中心】—— 新增芯片只在这里加！
# ==============================================
CHIP_CONFIGS = {
    'yorktown':{
        "single_err": 1e-3,  # SX门误差 (median)
        "two_err": 4e-2,  # CZ门误差 (median)
        "t1_us": 40,  # T1 (μs)
        "t2_us": 40,  # T2 (μs)
        "readout_err": 8e-2,  # 测量误差 (median)
        "single_gate_ns": 50,  # 单比特门时间
        "two_gate_ns": 300,  # 双比特门时间
        "basis_gates": ['cx', 'id', 'rz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "heron_r1": {
        "single_err": 2.999e-4,  # SX门误差 (median)
        "two_err": 2.589e-3,  # CZ门误差 (median)
        "t1_us": 175.85,  # T1 (μs)
        "t2_us": 134.17,  # T2 (μs)
        "readout_err": 2.95e-2,  # 测量误差 (median)
        "single_gate_ns": 50,  # 单比特门时间
        "two_gate_ns": 300,  # 双比特门时间
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "heron_r2": {
        "single_err": 2.263e-4,  # SX门误差
        "two_err": 2.194e-3,  # CZ门误差
        "t1_us": 238.22,  # T1 (μs)
        "t2_us": 123.16,  # T2 (μs)
        "readout_err": 0.0117,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "heron_r3": {
        "single_err": 1.484e-4,  # SX门误差
        "two_err": 1.199e-3,  # CZ门误差
        "t1_us": 282.14,  # T1 (μs)
        "t2_us": 327.59,  # T2 (μs)
        "readout_err": 5.25e-3,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "nighthawk_r1": {
        "single_err": 2.38e-4,  # SX门误差
        "two_err": 2.594e-3,  # CZ门误差
        "t1_us": 357.64,  # T1 (μs)
        "t2_us": 273.63,  # T2 (μs)
        "readout_err": 2.14e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "eagle_r3": {
        "single_err": 2.594e-4,  # SX门误差
        "two_err": 7.782e-3,  # ECR门误差
        "t1_us": 241.79,  # T1 (μs)
        "t2_us": 138.99,  # T2 (μs)
        "readout_err": 3.00e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['ecr', 'id', 'rz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    'toronto':{
        "single_err": 5.0318e-4,  # SX门误差
        "two_err": 8.94e-3,  # ECR门误差
        "t1_us": 112.79,  # T1 (μs)
        "t2_us": 153.99,  # T2 (μs)
        "readout_err": 5.00e-2,  # 测量误差
        "single_gate_ns": 568,
        "two_gate_ns": 6000,
        "basis_gates": [ 'id', 'rz', 'sx', 'x', 'cx'],
        "coupling_map": None,
        'map': [[0, 1], [1, 0], [1, 2], [1, 4], [2, 1], [2, 3], [3, 2], [3, 5], [4, 1], [4, 7], [5, 3], [5, 8], [6, 7], [7, 4], [7, 6], [7, 10], [8, 5], [8, 9], [8, 11], [9, 8], [10, 7], [10, 12], [11, 8], [11, 14], [12, 10], [12, 13], [12, 15], [13, 12], [13, 14], [14, 11], [14, 13], [14, 16], [15, 12], [15, 18], [16, 14], [16, 19], [17, 18], [18, 15], [18, 17], [18, 21], [19, 16], [19, 20], [19, 22], [20, 19], [21, 18], [21, 23], [22, 19], [22, 25], [23, 21], [23, 24], [24, 23], [24, 25], [25, 22], [25, 24], [25, 26], [26, 25]]
    }
}

# print(f"\n\n可选择芯片列表：{list(CHIP_CONFIGS.keys())}\n\n")
def get_chip_config(chip_name: str):
    if chip_name not in CHIP_CONFIGS:
        raise ValueError(f"芯片 {chip_name} 不存在！可选项：{list(CHIP_CONFIGS.keys())}")

    return copy.deepcopy(CHIP_CONFIGS[chip_name])


def create_noise_model(chip_name: str):
    """
    根据芯片名自动创建带对应噪声的 NoiseModel
    """
    cfg = get_chip_config(chip_name)

    # 单位转换
    t1 = cfg["t1_us"] * 1e-6
    t2 = cfg["t2_us"] * 1e-6

    # 噪声模型
    noise_model = NoiseModel()

    # ============== 单比特门噪声 ==============
    # 基础单比特门噪声
    err_single = depolarizing_error(cfg["single_err"], 1)
    thermal_single = thermal_relaxation_error(t1, t2, cfg["single_gate_ns"]* 1e-9)
    single_error = err_single.compose(thermal_single)

    single_qubit_gates = [ gate for gate in cfg["basis_gates"] if gate not in TWO_QUBIT_GATES and gate != 'cx']
    if single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(single_error, single_qubit_gates)

    # ============== 双比特门噪声 ==============
    two_qubit_gates = [gate for gate in cfg["basis_gates"] if gate in TWO_QUBIT_GATES ]
    # 基础双比特门噪声
    if 'map' in cfg:
        cx_pairs = cfg['map']
    else:
        cx_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    for pair in cx_pairs:
        err = cfg['two_err']
        length = cfg['two_gate_ns'] * 1e-9

        t1_c = t1
        t2_c = t2
        t1_t = t1
        t2_t = t2

        # 双比特弛豫噪声
        relax_qc = thermal_relaxation_error(t1_c, t2_c, length)
        relax_qt = thermal_relaxation_error(t1_t, t2_t, length)
        relax_total = relax_qc.expand(relax_qt)

        # 双比特去极化
        depol_total = depolarizing_error(err, 2)

        # 合并
        total_err = depol_total.compose(relax_total)
        noise_model.add_quantum_error(total_err, two_qubit_gates, pair)

    # ============== 测量噪声 ==============
    read_err = ReadoutError([
        [1 - cfg["readout_err"], cfg["readout_err"]],
        [cfg["readout_err"], 1 - cfg["readout_err"]]
    ])
    noise_model.add_all_qubit_readout_error(read_err)
    noise_model.add_basis_gates(cfg['basis_gates'])

    return noise_model