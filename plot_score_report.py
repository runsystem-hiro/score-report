import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge

# 日本語フォント設定
import matplotlib as mpl
plt.rcParams['font.family'] = 'BIZ UDGothic'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# CSVファイルからデータ読み込み
df = pd.read_csv('score_report_final.csv')
df['date'] = pd.to_datetime(df['date'])

# 累積学習時間の計算
df['cumulative_study_hours'] = df['study_hours'].cumsum()
df['cumulative_correct'] = df['correct_answers'].cumsum()

# プロットの設定
plt.style.use('ggplot')
fig = plt.figure(figsize=(15, 18))  # 高さを少し削減

# より効率的なGridSpecの設定
gs = GridSpec(6, 2, figure=fig, height_ratios=[1, 0.7, 1, 1, 0.7, 1.2])

main_color = '#3498db'
sub_colors = ['#e74c3c', '#2ecc71', '#f39c12']

# 1. 全体正解率の推移
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['date'], df['accuracy_per'], marker='o', markersize=8, color=main_color, linewidth=2.5)

ax1.set_title('全体正解率の推移', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('正解率 (%)', fontsize=11)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
for x, y in zip(df['date'], df['accuracy_per']):
    ax1.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

# 目標ラインを追加
ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7)
ax1.text(df['date'].iloc[0], 72, '目標: 70%', color='red', fontweight='bold')

# 2. 成績分布（レーダーチャート）
ax2 = fig.add_subplot(gs[0, 1], polar=True)
categories = ['ストラテジ系', 'マネジメント系', 'テクノロジ系']
N = len(categories)
latest_values = df.iloc[-1][['strategy_per', 'management_per', 'technology_per']].values
previous_values = df.iloc[-2][['strategy_per', 'management_per', 'technology_per']].values
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
latest_values = np.concatenate((latest_values, [latest_values[0]]))
previous_values = np.concatenate((previous_values, [previous_values[0]]))
ax2.plot(angles, latest_values, 'o-', linewidth=2, label='現在', color=main_color)
ax2.fill(angles, latest_values, alpha=0.25, color=main_color)
ax2.plot(angles, previous_values, 'o-', linewidth=2, label='前回', color='#e74c3c', alpha=0.7)
ax2.fill(angles, previous_values, alpha=0.1, color='#e74c3c')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax2.set_ylim(0, 100)
ax2.set_title('分野別成績分布', fontsize=13, fontweight='bold', pad=10)
ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=9)

# 3. 各カテゴリの改善度
ax3 = fig.add_subplot(gs[1, :])
category_names = ['ストラテジ系', 'マネジメント系', 'テクノロジ系']
improvements = latest_values[:-1] - previous_values[:-1]
colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
bars = ax3.bar(np.arange(len(category_names)), improvements, color=colors, alpha=0.7)
ax3.set_title('各分野の改善度', fontsize=13, fontweight='bold', pad=10)
ax3.set_xticks(np.arange(len(category_names)))
ax3.set_xticklabels(category_names, fontsize=11)
ax3.set_ylabel('変化量 (ポイント)', fontsize=11)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.grid(True, axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    sign = '+' if height > 0 else ''
    ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -5),
             f'{sign}{improvements[i]:.1f}', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=10, fontweight='bold')

# 4. 分野別学習進捗の推移と5. 自己学習時間の推移を一段目に
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df['date'], df['strategy_progress'], label='ストラテジ系', marker='o', color=sub_colors[0], markersize=6)
ax4.plot(df['date'], df['management_progress'], label='マネジメント系', marker='o', color=sub_colors[1], markersize=6)
ax4.plot(df['date'], df['technology_progress'], label='テクノロジ系', marker='o', color=sub_colors[2], markersize=6)
ax4.set_title('分野別学習進捗の推移', fontsize=13, fontweight='bold', pad=10)
ax4.set_ylabel('進捗率 (%)', fontsize=11)
ax4.set_ylim(0, 100)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(df['date'].dt.strftime('%Y-%m-%d'), df['study_hours'], color=main_color, alpha=0.7)
ax5.set_title('自己学習時間の推移', fontsize=13, fontweight='bold', pad=10)
ax5.set_ylabel('学習時間（時間）', fontsize=11)
ax5.tick_params(axis='x', rotation=30, labelsize=9)
ax5.grid(True, axis='y', alpha=0.3)

# 6. 累積正解数と累積学習時間（二軸グラフ）と7. 学習効率グラフを二段目に
ax6 = fig.add_subplot(gs[3, 0])
color1 = '#3498db'
color2 = '#e74c3c'

line1 = ax6.plot(df['date'], df['cumulative_correct'], marker='o', color=color1, linewidth=2, label='累積正解数', markersize=6)
ax6.set_ylabel('累積正解数', color=color1, fontsize=11)
ax6.tick_params(axis='y', labelcolor=color1)

ax6_2 = ax6.twinx()
line2 = ax6_2.plot(df['date'], df['cumulative_study_hours'], marker='s', color=color2, linewidth=2, label='累積学習時間', markersize=6)
ax6_2.set_ylabel('累積学習時間 (時間)', color=color2, fontsize=11)
ax6_2.tick_params(axis='y', labelcolor=color2)

# 凡例を結合
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, loc='upper left', fontsize=9)

ax6.set_title('累積正解数と累積学習時間', fontsize=13, fontweight='bold', pad=10)
ax6.grid(True, alpha=0.3)
ax6.tick_params(axis='x', rotation=30, labelsize=9)

# 7. 学習効率グラフ（正解数/学習時間）
ax7 = fig.add_subplot(gs[3, 1])
# 学習効率 = 正解数/学習時間
efficiency = []
for i in range(len(df)):
    if df.iloc[i]['study_hours'] > 0:
        efficiency.append(df.iloc[i]['correct_answers'] / df.iloc[i]['study_hours'])
    else:
        efficiency.append(0)

df['efficiency'] = efficiency
ax7.bar(df['date'].dt.strftime('%Y-%m-%d'), df['efficiency'], color='#2ecc71', alpha=0.7)
ax7.set_title('学習効率（正解数/学習時間）', fontsize=13, fontweight='bold', pad=10)
ax7.set_ylabel('効率（正解/時間）', fontsize=11)
ax7.tick_params(axis='x', rotation=30, labelsize=9)
ax7.grid(True, axis='y', alpha=0.3)

# 8. KPIゲージ（目標70%に対する達成度）を9. 分野別改善ニーズと同じ行に
ax8 = fig.add_subplot(gs[4, 0], aspect='equal')
ax8.set_title('KPI達成度（目標:70%）', fontsize=13, fontweight='bold', pad=10)

# ゲージの描画
target = 70
latest = df.iloc[-1]['accuracy_per']
achievement_rate = min(latest / target, 1.0)  # 達成率（最大1.0）

# 外側のゲージ（グレー）
wedge_outer = Wedge((0.5, 0.5), 0.4, 0, 360, width=0.1, facecolor='lightgray', edgecolor='none')
ax8.add_patch(wedge_outer)

# 内側のゲージ（カラー）- 達成度に応じた色と角度
gauge_angle = 360 * achievement_rate
gauge_color = '#e74c3c' if achievement_rate < 0.5 else ('#f39c12' if achievement_rate < 0.8 else '#2ecc71')
wedge_inner = Wedge((0.5, 0.5), 0.4, 0, gauge_angle, width=0.1, facecolor=gauge_color, edgecolor='none')
ax8.add_patch(wedge_inner)

# テキスト表示
ax8.text(0.5, 0.5, f'{latest:.1f}%\n達成率: {achievement_rate*100:.1f}%', 
         horizontalalignment='center', verticalalignment='center', fontsize=13, fontweight='bold')
ax8.text(0.5, 0.2, f'目標: {target}%', horizontalalignment='center', verticalalignment='center', fontsize=11)

# 軸の設定
ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)
ax8.axis('off')

# 9. 分野別改善ニーズ（弱点分析）
ax9 = fig.add_subplot(gs[4, 1])
categories = ['ストラテジ系', 'マネジメント系', 'テクノロジ系']
latest_values = df.iloc[-1][['strategy_per', 'management_per', 'technology_per']].values
target_gap = [max(target - val, 0) for val in latest_values]  # 目標との差分

# 改善ニーズの強さで色を変える
colors = ['#e74c3c' if gap > 30 else ('#f39c12' if gap > 15 else '#2ecc71') for gap in target_gap]

bars = ax9.bar(categories, target_gap, color=colors, alpha=0.7)
ax9.set_title('分野別改善ニーズ（目標70%との差）', fontsize=13, fontweight='bold', pad=10)
ax9.set_ylabel('改善ニーズ（ポイント）', fontsize=11)
ax9.grid(True, axis='y', alpha=0.3)

# バーの上に数値表示
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0:
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 10. 総合成績のサマリーは最後に配置
ax10 = fig.add_subplot(gs[5, :])
ax10.axis('off')
headers = ['報告日', '問題数', '正解数', '正解率', '評価']
table_data = [headers]
for i in range(len(df)):
    row = [df.iloc[i]['date'].strftime('%Y-%m-%d'),
           f"{int(df.iloc[i]['total_questions'])}",
           f"{int(df.iloc[i]['correct_answers'])}",
           f"{df.iloc[i]['accuracy_per']}%",
           f"{df.iloc[i]['grade']}"]
    table_data.append(row)
table = ax10.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2]*5)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)  # 少し高さを調整
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(color='white', fontweight='bold')
if df.iloc[-1]['accuracy_per'] > df.iloc[-2]['accuracy_per']:
    for i in range(len(headers)):
        table[(len(df), i)].set_facecolor('#e8f4f8')

# 全体タイトルと改善メッセージ
plt.suptitle('試験成績レポート', fontsize=18, fontweight='bold', y=0.98)
overall_improvement = df.iloc[-1]['accuracy_per'] - df.iloc[-2]['accuracy_per']
improvement_message = f"前回と比較して全体成績が{overall_improvement:.1f}ポイント"
improvement_message += "向上しました！" if overall_improvement > 0 else "低下しています。"
fig.text(0.5, 0.02, improvement_message, ha='center', fontsize=12, fontweight='bold',
         color='#2ecc71' if overall_improvement > 0 else '#e74c3c')

# 今後の勉強アドバイス
weakest_area_idx = np.argmin(latest_values)
weakest_area = categories[weakest_area_idx]
advice = f"アドバイス: {weakest_area}の学習強化が最も効果的です。"
fig.text(0.5, 0.01, advice, ha='center', fontsize=11, fontweight='bold', color='#3498db')

plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
plt.savefig('score_report.png', dpi=300, bbox_inches='tight')
print("[INFO] score_report.png を出力しました")
# plt.show()