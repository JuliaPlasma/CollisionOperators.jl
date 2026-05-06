const V_MIN, V_MAX = -6.0, 6.0
const I_MIN, I_MAX = -4.0, 4.0
const P_DEG = 2                # B-spline 多项式阶数
const K_REG = 1                # 内部结点正则性
const N_ELEM = 20              # 每维内层元素数 (在 [I_MIN, I_MAX] 内). 外层各加 1 元素覆盖到 V_MIN/V_MAX, 故总数 N_ELEM+2 = 22
const N_PARTICLES = 10_000       # 粒子数 (O(N²) 碰撞)
const σ₁, σ₂ = 4/3, 0.5       # 各向异性高斯: v₁ ~ N(0,σ₁²), v₂ ~ N(0,σ₂²); 3σ₁ = I_MAX, 比例 σ₁:σ₂ = 8:3
const DT = 0.001
const N_STEPS = 400
const N_QUAD = 6               # 积分点数/维
