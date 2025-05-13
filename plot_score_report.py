
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 日本語フォント設定
import matplotlib as mpl
plt.rcParams['font.family'] = 'BIZ UDGothic'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# CSVファイルからデータ読み込み
df = pd.read_csv('score_report_final.csv')
df['date'] = pd.to_datetime(df['date'])

# プロットの設定
plt.style.use('ggplot')
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(4, 2, figure=fig)

main_color = '#3498db'
sub_colors = ['#e74c3c', '#2ecc71', '#f39c12']

# 1. 全体正解率の推移
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['date'], df['accuracy_per'], marker='o', markersize=10, color=main_color, linewidth=3)
ax1.set_title('全体正解率の推移', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('正解率 (%)', fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
for x, y in zip(df['date'], df['accuracy_per']):
    ax1.annotate(f'{y}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')

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
ax2.set_xticklabels(categories, fontsize=12)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax2.set_ylim(0, 100)
ax2.set_title('分野別成績分布', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 3. 各カテゴリの改善度
ax3 = fig.add_subplot(gs[1, :])
category_names = ['ストラテジ系', 'マネジメント系', 'テクノロジ系']
improvements = latest_values[:-1] - previous_values[:-1]
colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
bars = ax3.bar(np.arange(len(category_names)), improvements, color=colors, alpha=0.7)
ax3.set_title('各分野の改善度', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(np.arange(len(category_names)))
ax3.set_xticklabels(category_names, fontsize=12)
ax3.set_ylabel('変化量 (ポイント)', fontsize=12)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.grid(True, axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    sign = '+' if height > 0 else ''
    ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -5),
             f'{sign}{improvements[i]:.1f}', ha='center', va='bottom' if height > 0 else 'top',
             fontsize=11, fontweight='bold')

# 4. 分野別学習進捗の推移
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df['date'], df['strategy_progress'], label='ストラテジ系', marker='o', color=sub_colors[0])
ax4.plot(df['date'], df['management_progress'], label='マネジメント系', marker='o', color=sub_colors[1])
ax4.plot(df['date'], df['technology_progress'], label='テクノロジ系', marker='o', color=sub_colors[2])
ax4.set_title('分野別学習進捗の推移', fontsize=14, fontweight='bold', pad=15)
ax4.set_ylabel('進捗率 (%)', fontsize=12)
ax4.set_ylim(0, 100)
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 5. 自己学習時間の推移
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(df['date'].dt.strftime('%Y-%m-%d'), df['study_hours'], color=main_color, alpha=0.7)
ax5.set_title('自己学習時間の推移', fontsize=14, fontweight='bold', pad=15)
ax5.set_ylabel('学習時間（時間）', fontsize=12)
ax5.set_xlabel('日付')
ax5.grid(True, axis='y', alpha=0.3)

# 6. 総合成績のサマリー
ax6 = fig.add_subplot(gs[3, :])
ax6.axis('off')
headers = ['報告日', '問題数', '正解数', '正解率', '評価']
table_data = [headers]
for i in range(len(df)):
    row = [df.iloc[i]['date'].strftime('%Y-%m-%d'),
           f"{int(df.iloc[i]['total_questions'])}",
           f"{int(df.iloc[i]['correct_answers'])}",
           f"{df.iloc[i]['accuracy_per']}%",
           f"{df.iloc[i]['grade']}"]
    table_data.append(row)
table = ax6.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2]*5)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(color='white', fontweight='bold')
if df.iloc[-1]['accuracy_per'] > df.iloc[-2]['accuracy_per']:
    for i in range(len(headers)):
        table[(len(df), i)].set_facecolor('#e8f4f8')

# 全体タイトルと改善メッセージ
plt.suptitle('試験成績レポート', fontsize=20, fontweight='bold', y=0.99)
overall_improvement = df.iloc[-1]['accuracy_per'] - df.iloc[-2]['accuracy_per']
improvement_message = f"前回と比較して全体成績が{overall_improvement:.1f}ポイント"
improvement_message += "向上しました！" if overall_improvement > 0 else "低下しています。"
fig.text(0.5, 0.03, improvement_message, ha='center', fontsize=13, fontweight='bold',
         color='#2ecc71' if overall_improvement > 0 else '#e74c3c')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('score_report.png', dpi=300, bbox_inches='tight')
# plt.show()
