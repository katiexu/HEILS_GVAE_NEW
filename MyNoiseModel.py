from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError
)

# ==============================================
# 【芯片噪声配置中心】—— 新增芯片只在这里加！
# ==============================================
CHIP_CONFIGS = {
    "heron_r1": {
        "num_qubits": 4,  # 测试用4量子比特，实际为133
        "single_err": 2.999e-4,  # SX门误差 (median)
        "two_err": 2.589e-3,  # CZ门误差 (median)
        "t1_us": 175.85,  # T1 (μs)
        "t2_us": 134.17,  # T2 (μs)
        "readout_err": 2.95e-2,  # 测量误差 (median)
        "single_gate_ns": 50,  # 单比特门时间
        "two_gate_ns": 300,  # 双比特门时间
        "u3_gate_ns": 150,  # U3门时间（3倍单比特门时间）
        "cu3_gate_ns": 450,  # CU3门时间（1.5倍双比特门时间）
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure', 'u3', 'cu3'],
        "coupling_map": None
    },
    "heron_r2": {
        "num_qubits": 4,  # 测试用4量子比特，实际为156
        "single_err": 2.263e-4,  # SX门误差
        "two_err": 2.194e-3,  # CZ门误差
        "t1_us": 238.22,  # T1 (μs)
        "t2_us": 123.16,  # T2 (μs)
        "readout_err": 0.0117,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "u3_gate_ns": 150,
        "cu3_gate_ns": 450,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure', 'u3', 'cu3'],
        "coupling_map": None
    },
    "heron_r3": {
        "num_qubits": 4,  # 测试用4量子比特，实际为156
        "single_err": 1.484e-4,  # SX门误差
        "two_err": 1.199e-3,  # CZ门误差
        "t1_us": 282.14,  # T1 (μs)
        "t2_us": 327.59,  # T2 (μs)
        "readout_err": 5.25e-3,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "u3_gate_ns": 150,
        "cu3_gate_ns": 450,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure', 'u3', 'cu3'],
        "coupling_map": None
    },
    "nighthawk_r1": {
        "num_qubits": 4,  # 测试用4量子比特，实际为120
        "single_err": 2.38e-4,  # SX门误差
        "two_err": 2.594e-3,  # CZ门误差
        "t1_us": 357.64,  # T1 (μs)
        "t2_us": 273.63,  # T2 (μs)
        "readout_err": 2.14e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "u3_gate_ns": 150,
        "cu3_gate_ns": 450,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'sx', 'x', 'measure', 'u3', 'cu3'],
        "coupling_map": None
    },
    "eagle_r3": {
        "num_qubits": 4,  # 测试用4量子比特，实际为127
        "single_err": 2.594e-4,  # SX门误差
        "two_err": 7.782e-3,  # ECR门误差
        "t1_us": 241.79,  # T1 (μs)
        "t2_us": 138.99,  # T2 (μs)
        "readout_err": 3.00e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "u3_gate_ns": 150,
        "cu3_gate_ns": 450,
        "basis_gates": ['ecr', 'id', 'rz', 'sx', 'x', 'measure', 'u3', 'cu3'],
        "coupling_map": None
    },
}

print(f"\n\n可选择芯片列表：{list(CHIP_CONFIGS.keys())}\n\n")
def create_noise_model(chip_name: str):
    """
    根据芯片名自动创建带对应噪声的 NoiseModel
    """
    if chip_name not in CHIP_CONFIGS:
        raise ValueError(f"芯片 {chip_name} 不存在！可选项：{list(CHIP_CONFIGS.keys())}")

    cfg = CHIP_CONFIGS[chip_name]

    # 单位转换
    t1_ns = cfg["t1_us"] * 1000
    t2_ns = cfg["t2_us"] * 1000

    # 噪声模型
    noise_model = NoiseModel()

    # ============== 单比特门噪声 ==============
    # 基础单比特门噪声
    err_single = depolarizing_error(cfg["single_err"], 1)
    thermal_single = thermal_relaxation_error(t1_ns, t2_ns, cfg["single_gate_ns"])
    single_error = err_single.compose(thermal_single)

    # 识别基础单比特门（排除U3、双比特门和measure）
    basic_single_qubit_gates = [
        g for g in cfg["basis_gates"]
        if g not in ['u3', 'cu3', 'cx', 'cz', 'rzz', 'ecr', 'measure'] and g != 'measure'
    ]

    if basic_single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(single_error, basic_single_qubit_gates)

    # ============== 特殊的 U3 门噪声 ==============
    if "u3" in cfg["basis_gates"]:
        # U3 是复杂单比特门，有更高的误差和门时间
        err_u3 = depolarizing_error(cfg["single_err"] * 3, 1)  # 误差是基础门的3倍
        thermal_u3 = thermal_relaxation_error(t1_ns, t2_ns, cfg["u3_gate_ns"])
        u3_error = err_u3.compose(thermal_u3)
        noise_model.add_all_qubit_quantum_error(u3_error, ["u3"])

    # ============== 双比特门噪声 ==============
    # 基础双比特门噪声
    err_two = depolarizing_error(cfg["two_err"], 2)
    thermal_two = thermal_relaxation_error(t1_ns, t2_ns, cfg["two_gate_ns"])
    two_qubit_error = err_two.compose(thermal_two)

    # 处理不同类型的双比特门
    two_qubit_gate_types = {}

    # 检查芯片支持的双比特门类型
    for gate in cfg["basis_gates"]:
        if gate in ['cx', 'cz', 'rzz', 'ecr']:
            two_qubit_gate_types[gate] = True

    # 为每种双比特门添加噪声
    for gate_type in two_qubit_gate_types:
        if gate_type == 'ecr':
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ["ecr"])
        elif gate_type == 'cz':
            # CZ 和 RZZ 使用相同的噪声模型
            cz_gates = [g for g in ['cz', 'rzz'] if g in cfg["basis_gates"]]
            if cz_gates:
                noise_model.add_all_qubit_quantum_error(two_qubit_error, cz_gates)
        elif gate_type == 'cx':
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])

    # ============== 特殊的 CU3 门噪声 ==============
    if "cu3" in cfg["basis_gates"]:
        # CU3 是复杂的受控门，有更高的误差和门时间
        err_cu3 = depolarizing_error(cfg["two_err"] * 2, 2)  # 误差是基础双比特门的2倍
        thermal_cu3 = thermal_relaxation_error(t1_ns, t2_ns, cfg["cu3_gate_ns"])
        cu3_error = err_cu3.compose(thermal_cu3)
        noise_model.add_all_qubit_quantum_error(cu3_error, ["cu3"])

    # ============== 测量噪声 ==============
    read_err = ReadoutError([
        [1 - cfg["readout_err"], cfg["readout_err"]],
        [cfg["readout_err"], 1 - cfg["readout_err"]]
    ])
    noise_model.add_all_qubit_readout_error(read_err)

    return noise_model