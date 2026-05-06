import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 全局字体与绘图格式设置 (IEEE/ISSCC 规范)
# ==========================================
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stix'

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10

sns.set_theme(style="white", rc={
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in"
})

# ==========================================
# 2. 核心算法：倒序对数-线性插值找点
# ==========================================
def find_last_crossing_log_linear(freqs, gains, target_gain):
    """
    针对X轴为对数、Y轴为线性的数据，从右向左（高频向低频）寻找交点。
    这样可以确保找到频率最大的穿越点。
    """
    # 从倒数第二个点开始向前遍历
    for i in range(len(gains) - 2, -1, -1):
        if (gains[i] >= target_gain and gains[i+1] <= target_gain) or \
           (gains[i] <= target_gain and gains[i+1] >= target_gain):
            
            f1, f2 = freqs[i], freqs[i+1]
            g1, g2 = gains[i], gains[i+1]
            
            log_f1 = np.log10(f1)
            log_f2 = np.log10(f2)
            
            # 线性插值比例
            fraction = (target_gain - g1) / (g2 - g1)
            log_fc = log_f1 + fraction * (log_f2 - log_f1)
            
            return 10**log_fc, target_gain
            
    return None, None

def format_frequency(f):
    if f >= 1e9:
        return f"{f/1e9:.2f} GHz"
    elif f >= 1e6:
        return f"{f/1e6:.2f} MHz"
    elif f >= 1e3:
        return f"{f/1e3:.2f} kHz"
    else:
        return f"{f:.2f} Hz"

# ==========================================
# 3. 数据读取与处理 (包含基准增益智能识别)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'tia_test.csv')

try:
    df = pd.read_csv(file_path)
    freq = df.iloc[:, 0].values.astype(float)
    gain = df.iloc[:, 1].values.astype(float)
except Exception as e:
    print(f"读取数据失败！请确保文件存在。\n路径: {file_path}\n错误信息: {e}")
    exit()

# === 关键修复：计算中频平坦区增益 (Mid-band Gain) ===
# 策略：取最高增益向下 15dB 范围内的所有点，计算其中位数。
# 因为在对数频率轴上，平坦区占据的数据点数量呈压倒性优势，中位数能完美避开极少量的低频尖峰点。
top_gains = gain[gain > (np.max(gain) - 15)]
ref_gain = np.median(top_gains) 

# 设置目标增益
target_3db = ref_gain - 3.0
target_0db = 0.0

# 从右向左寻找频率最大的交叉点
f_3db, g_3db = find_last_crossing_log_linear(freq, gain, target_3db)
f_0db, g_0db = find_last_crossing_log_linear(freq, gain, target_0db)

# ==========================================
# 4. 绘图配置与实现
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(freq, gain, color='#1A2B4C', linewidth=1.8, label='AC Response')
ax.set_xscale('log')

ax.set_title("TIA AC Response", fontsize=TITLE_SIZE, fontfamily='Times New Roman')
ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_SIZE, fontfamily='Times New Roman')
ax.set_ylabel("Gain (dB)", fontsize=LABEL_SIZE, fontfamily='Times New Roman')

ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, length=5, width=1.0, direction='in')
ax.tick_params(axis='both', which='minor', length=3, width=0.5, direction='in')

ax.grid(True, which='major', linestyle='--', color='#CCCCCC', linewidth=0.8, alpha=0.7)
ax.grid(True, which='minor', linestyle=':', color='#E0E0E0', linewidth=0.5, alpha=0.5)

# ==========================================
# 5. 关键点标注与渲染
# ==========================================
marker_color = '#DC143C'

if f_3db is not None:
    ax.scatter(f_3db, g_3db, color=marker_color, s=50, zorder=5)
    text_3db = f"-3dB BW: {format_frequency(f_3db)}\n({g_3db:.2f} dB)"
    ax.annotate(text_3db, 
                xy=(f_3db, g_3db), 
                # 修改偏移量，向左下方偏移，避免高频下降沿的遮挡
                xytext=(-15, -35), 
                textcoords='offset points', 
                ha='right', va='top',
                fontsize=TICK_SIZE, color=marker_color, fontfamily='Times New Roman',
                arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.2, alpha=0.8))

if f_0db is not None:
    ax.scatter(f_0db, g_0db, color=marker_color, s=50, zorder=5)
    text_0db = f"Unity Gain: {format_frequency(f_0db)}\n(0 dB)"
    ax.annotate(text_0db, 
                xy=(f_0db, g_0db), 
                xytext=(15, 15), 
                textcoords='offset points',
                ha='left', va='bottom',
                fontsize=TICK_SIZE, color=marker_color, fontfamily='Times New Roman',
                arrowprops=dict(arrowstyle='->', color=marker_color, lw=1.2, alpha=0.8))

# ==========================================
# 6. 添加专属水印
# ==========================================
ax.text(0.97, 0.96, '@CICC1008916', 
        transform=ax.transAxes,    
        fontsize=TICK_SIZE,        
        color='black',             
        alpha=0.5,                 
        ha='right', va='top',      
        fontfamily='Times New Roman',
        zorder=10)                 

# ==========================================
# 7. 导出与保存
# ==========================================
output_filename = os.path.join(current_dir, 'tia_test.png')
plt.savefig(output_filename, 
            dpi=600,               
            bbox_inches='tight',   
            facecolor='white',     
            transparent=False)

print(f"基准中频增益计算为: {ref_gain:.2f} dB")
if f_3db:
    print(f"成功找到 -3dB 带宽点: {format_frequency(f_3db)}")
print(f"图表已成功生成并保存为: {output_filename}")