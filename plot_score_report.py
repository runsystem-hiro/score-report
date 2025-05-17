import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from matplotlib.font_manager import FontProperties
import sys
from datetime import timedelta
import math

def compute_smart_ylim(min_val, max_val, 
                       lower_margin_ratio=1.5, 
                       upper_margin_ratio=0.2, 
                       ymin_limit=None, 
                       ymax_limit=None):
    """
    グラフのY軸範囲を自動的に決定するユーティリティ関数。

    Parameters:
    - min_val (float): 改善値などの最小値
    - max_val (float): 最大値
    - lower_margin_ratio (float): 負の値への余白倍率（例：1.5 → 値の1.5倍余白）
    - upper_margin_ratio (float): 正の値への余白倍率（例：0.2 → 20%余白）
    - ymin_limit (float): 下限値の最大値（例：-25など）
    - ymax_limit (float): 上限値の最大値（例：100など）

    Returns:
    - (ymin, ymax): Y軸の範囲タプル（整数）
    """
    ymin = min_val - abs(min_val) * lower_margin_ratio if min_val < 0 else 0
    ymax = max_val + abs(max_val) * upper_margin_ratio if max_val > 0 else 0

    if ymin_limit is not None:
        ymin = min(ymin, ymin_limit)
    if ymax_limit is not None:
        ymax = max(ymax, ymax_limit)

    return math.floor(ymin), math.ceil(ymax)

# フォントファイルのパスと存在確認
font_path = "/workspace/fonts/BIZUDGothic-Regular.ttf"
try:
    if not os.path.exists(font_path):
        print(f"[警告] フォントファイルが見つかりません: {font_path}")
        print("システムにインストールされている日本語フォントを探します...")
        # 代替フォントを探す
        import matplotlib.font_manager as fm
        fonts = [f.name for f in fm.fontManager.ttflist if 'gothic' in f.name.lower() or 'meiryo' in f.name.lower() or 'yu' in f.name.lower()]
        if fonts:
            print(f"利用可能な日本語フォント: {', '.join(fonts[:5])}")
            font_prop = FontProperties(family=fonts[0])
        else:
            print("日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
            font_prop = FontProperties()
    else:
        font_prop = FontProperties(fname=font_path)
        print(f"[INFO] フォントを読み込みました: {font_path}")
except Exception as e:
    print(f"[エラー] フォント設定中にエラーが発生しました: {e}")
    font_prop = FontProperties()

# 日本語フォント設定
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 全てのテキスト要素に日本語フォントを適用する関数
def set_font_for_all_texts(fig):
    for ax in fig.get_axes():
        texts = ax.get_xticklabels() + ax.get_yticklabels()
        for text in texts:
            text.set_fontproperties(font_prop)
        
        # タイトルにもフォント設定
        title = ax.get_title()
        if title:
            ax.set_title(title, fontproperties=font_prop)
        
        # 軸ラベルにもフォント設定
        xlabel = ax.get_xlabel()
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=font_prop)
        
        ylabel = ax.get_ylabel()
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=font_prop)
        
        # 凡例にもフォント設定
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)

# CSVファイル読み込みエラーハンドリング
try:
    df = pd.read_csv('score_report_final.csv')
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("[エラー] CSVファイルが見つかりません: resources/score_report_final.csv")
    sys.exit(1)
except Exception as e:
    print(f"[エラー] データ読み込み中にエラーが発生しました: {e}")
    sys.exit(1)

# 累積学習時間の計算
df['cumulative_study_hours'] = df['study_hours'].cumsum()
df['cumulative_correct'] = df['correct_answers'].cumsum()

# 日付フォーマットの調整（2025年を省略）
def format_date_without_year(date):
    return date.strftime('%m-%d')

# x軸ラベルを整形
date_labels = [format_date_without_year(d) for d in df['date']]

# プロットの設定
plt.style.use('ggplot')
fig = plt.figure(figsize=(15, 20))  # 高さを大幅に増加

# より効率的なGridSpecの設定（高さ比率を調整）
gs = GridSpec(6, 2, figure=fig, height_ratios=[3, 4, 3, 3, 4, 2])
plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.12, hspace=0.3)

main_color = '#3498db'
sub_colors = ['#e74c3c', '#2ecc71', '#f39c12']

# 1. 全体正解率の推移
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['date'], df['accuracy_per'], marker='o', markersize=8, color=main_color, linewidth=2.5)

ax1.set_title('全体正解率の推移', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax1.set_ylabel('正解率 (%)', fontsize=12, fontproperties=font_prop)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# x軸のラベル設定改善
ax1.set_xticks(df['date'])
ax1.set_xticklabels(date_labels)

# データポイントの注釈位置を調整
for x, y, label in zip(df['date'], df['accuracy_per'], df['accuracy_per']):
    ax1.annotate(f'{label}%', (x, y), textcoords="offset points", xytext=(0, 10), 
                 ha='center', fontsize=11, fontweight='bold', fontproperties=font_prop)

# 目標ラインを追加
ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7)
ax1.text(df['date'].iloc[0], 72, '目標: 70%', color='red', fontweight='bold', fontsize=12, fontproperties=font_prop)

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
ax2.set_xticklabels(categories, fontsize=12, fontproperties=font_prop)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax2.set_ylim(0, 100)
ax2.set_title('分野別成績分布', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=11, prop=font_prop)

# 3. 各カテゴリの改善度
ax3 = fig.add_subplot(gs[1, :])
category_names = ['ストラテジ系', 'マネジメント系', 'テクノロジ系']
improvements = latest_values[:-1] - previous_values[:-1]
colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
bars = ax3.bar(np.arange(len(category_names)), improvements, color=colors, alpha=0.7)

# 高さを調整（非対称に）
min_val = improvements.min()
max_val = improvements.max()

# ymin = math.floor(min_val - abs(min_val) * 1.5) if min_val < 0 else 0
# ymax = math.ceil(max_val + abs(max_val) * 0.2) if max_val > 0 else 0

ymin, ymax = compute_smart_ylim(min_val, max_val, ymin_limit=-5)

ax3.set_ylim(ymin, ymax)

ax3.set_title('各分野の改善度', fontsize=14, fontweight='bold', pad=25, fontproperties=font_prop)
ax3.set_xticks(np.arange(len(category_names)))
ax3.set_xticklabels(category_names, fontsize=12, fontproperties=font_prop)
ax3.set_ylabel('変化量 (ポイント)', fontsize=12, fontproperties=font_prop)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.grid(True, axis='y', alpha=0.3)

# ラベル表示（±に対応）
for i, bar in enumerate(bars):
    height = bar.get_height()
    offset = abs(height) * 0.07
    sign = '+' if height > 0 else ''
    if height > 0:
        y_pos = height - offset
        va = 'top'
    else:
        y_pos = height + offset
        va = 'bottom'

    ax3.text(bar.get_x() + bar.get_width() / 2., y_pos,
             f'{sign}{improvements[i]:.1f}',
             ha='center', va=va,
             fontsize=11, fontweight='bold', fontproperties=font_prop)

# 4. 分野別学習進捗の推移
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df['date'], df['strategy_progress'], label='ストラテジ系', marker='o', color=sub_colors[0], markersize=6)
ax4.plot(df['date'], df['management_progress'], label='マネジメント系', marker='o', color=sub_colors[1], markersize=6)
ax4.plot(df['date'], df['technology_progress'], label='テクノロジ系', marker='o', color=sub_colors[2], markersize=6)
ax4.set_title('分野別学習進捗の推移', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax4.set_ylabel('進捗率 (%)', fontsize=12, fontproperties=font_prop)
ax4.set_ylim(0, 100)
ax4.legend(loc='lower right', fontsize=11, prop=font_prop)
ax4.grid(True, alpha=0.3)

# x軸のラベル設定改善
ax4.set_xticks(df['date'])
ax4.set_xticklabels(date_labels)

# 5. 自己学習時間の推移 - 棒グラフの幅を調整
ax5 = fig.add_subplot(gs[2, 1])
# 棒グラフの幅を広げるために、0.6の値を使用（デフォルトは0.8）
ax5.bar(df['date'], df['study_hours'], color=main_color, alpha=0.7, width=5.0)
ax5.set_title('自己学習時間の推移', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax5.set_ylabel('学習時間（時間）', fontsize=12, fontproperties=font_prop)

# x軸のラベル設定改善
ax5.set_xticks(df['date'])
ax5.set_xticklabels(date_labels, rotation=30, ha='right')

# 棒グラフの表示数値の調整
ax5.grid(True, axis='y', alpha=0.3)

# 学習時間に値を表示
for i, v in enumerate(df['study_hours']):
    offset = v * 0.07
    ax5.text(df['date'].iloc[i], v - offset, f'{v:.1f}', ha='center', va='top', fontsize=11, fontweight='bold', fontproperties=font_prop)

# 6. 累積正解数と累積学習時間（二軸グラフ）
ax6 = fig.add_subplot(gs[3, 0])
color1 = '#3498db'
color2 = '#e74c3c'

line1 = ax6.plot(df['date'], df['cumulative_correct'], marker='o', color=color1, linewidth=2, label='累積正解数', markersize=6)
ax6.set_ylabel('累積正解数', color=color1, fontsize=12, fontproperties=font_prop)
ax6.tick_params(axis='y', labelcolor=color1)

ax6_2 = ax6.twinx()
line2 = ax6_2.plot(df['date'], df['cumulative_study_hours'], marker='s', color=color2, linewidth=2, label='累積学習時間', markersize=6)
ax6_2.set_ylabel('累積学習時間 (時間)', color=color2, fontsize=12, fontproperties=font_prop)
ax6_2.tick_params(axis='y', labelcolor=color2)

# 凡例を結合
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax6.legend(lines, labels, loc='upper left', fontsize=11, prop=font_prop)

ax6.set_title('累積正解数と累積学習時間', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax6.grid(True, alpha=0.3)

# x軸のラベル設定改善
ax6.set_xticks(df['date'])
ax6.set_xticklabels(date_labels, rotation=30, ha='right')

# 7. 学習効率グラフ（正解数/学習時間） - 棒グラフの幅を調整
ax7 = fig.add_subplot(gs[3, 1])

# 学習効率の計算
df['efficiency'] = 0.0  # 初期化
for i in range(len(df)):
    if df.iloc[i]['study_hours'] > 0:
        df.at[i, 'efficiency'] = df.iloc[i]['correct_answers'] / df.iloc[i]['study_hours']

# 棒グラフの幅を広げるために、0.4の値を使用
ax7.bar(df['date'], df['efficiency'], color='#2ecc71', alpha=0.7, width=5.0)
ax7.set_title('学習効率（正解数/学習時間）', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax7.set_ylabel('効率（正解/時間）', fontsize=12, fontproperties=font_prop)

# x軸のラベル設定改善
ax7.set_xticks(df['date'])
ax7.set_xticklabels(date_labels, rotation=30, ha='right')

ax7.grid(True, axis='y', alpha=0.3)

# 効率値を表示
for i, v in enumerate(df['efficiency']):
    if not np.isnan(v) and v > 0:
        offset = v * 0.07
        ax7.text(df['date'].iloc[i], v - offset, f'{v:.1f}', ha='center', va='top', fontsize=11, fontweight='bold', fontproperties=font_prop)

# 8. KPIゲージ（目標70%に対する達成度） - 12時から時計回りに変更
ax8 = fig.add_subplot(gs[4, 0], aspect='equal')
ax8.set_title('KPI正答率達成度（目標正答率:70%）', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)

# --- 設定値 ---
target = 70.0
latest = df.iloc[-1]['accuracy_per']
max_rate = 100 / target  # 例: 100 ÷ 70 = 1.428571...

# --- 達成率スケーリング（最大 142.9% = 360度）---
achievement_rate = latest / target
normalized_rate = min(achievement_rate, max_rate)
gauge_angle = 360 * (normalized_rate / max_rate)

# --- カラー設定 ---
if achievement_rate < 0.4:
    gauge_color = '#bdc3c7'  # 灰色：未達成
elif achievement_rate < 0.6:
    gauge_color = '#e74c3c'  # 赤：低水準
elif achievement_rate < 0.85:
    gauge_color = '#f39c12'  # オレンジ：中程度
elif achievement_rate < 1.0:
    gauge_color = '#bada55'  # 黄緑：惜しいが未達
elif achievement_rate < 1.2:
    gauge_color = '#2ecc71'  # 緑：目標達成
else:
    gauge_color = '#3498db'  # 青：超過達成

# --- 描画用角度 ---
start_angle = 90
end_angle = (start_angle - gauge_angle) % 360  # 時計回り表現（反時計回り描画を逆手に）

# --- ゲージ本体（オレンジなど） ---
wedge_inner = Wedge((0.5, 0.5), 0.4, end_angle, start_angle, width=0.1, facecolor=gauge_color, edgecolor='none')
ax8.add_patch(wedge_inner)

# --- 背景（灰色：未達成分） ---
if gauge_angle < 360:
    wedge_outer = Wedge((0.5, 0.5), 0.4, start_angle, end_angle, width=0.1, facecolor='lightgray', edgecolor='none')
    ax8.add_patch(wedge_outer)

# --- テキスト表示 ---
ax8.text(0.5, 0.5, f'正答:{latest:.1f}%\n達成度: {achievement_rate*100:.1f}%',
         ha='center', va='center', fontsize=14, fontweight='bold', fontproperties=font_prop)

# --- 軸設定 ---
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
ax9.set_title('分野別改善ニーズ（目標70%との差）', fontsize=14, fontweight='bold', pad=15, fontproperties=font_prop)
ax9.set_ylabel('改善ニーズ（ポイント）', fontsize=12, fontproperties=font_prop)
ax9.grid(True, axis='y', alpha=0.3)

# バーの上に数値表示
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0:
        offset = height * 0.07
        ax9.text(bar.get_x() + bar.get_width()/2., height - offset,
                f'{height:.1f}', ha='center', va='top', fontsize=11, 
                fontweight='bold', fontproperties=font_prop)

# 10. 総合成績のサマリー
ax10 = fig.add_subplot(gs[5, :])
ax10.axis('off')
headers = ['報告日', '問題数', '正解数', '正解率', '評価']
table_data = [headers]
for i in range(len(df)):
    # row = [format_date_without_year(df.iloc[i]['date']),
    row = [df.iloc[i]['date'].strftime('%Y-%m-%d'),           
           f"{int(df.iloc[i]['total_questions'])}",
           f"{int(df.iloc[i]['correct_answers'])}",
           f"{df.iloc[i]['accuracy_per']}%",
           f"{df.iloc[i]['grade']}"]
    table_data.append(row)
table = ax10.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2]*5)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)  # テーブルの高さを調整

# テーブルのセルに日本語フォントを適用
for i in range(len(table_data)):
    for j in range(len(headers)):
        cell = table[(i, j)]
        text = cell.get_text()
        text.set_fontproperties(font_prop)

# ヘッダー行の色設定
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(color='white', fontweight='bold')

# 最新データ行のハイライト
if df.iloc[-1]['accuracy_per'] > df.iloc[-2]['accuracy_per']:
    for i in range(len(headers)):
        table[(len(df), i)].set_facecolor('#e8f4f8')

# 全体タイトルと改善メッセージ
fig.suptitle("試験成績レポート", fontsize=18, fontweight='bold', y=0.965, fontproperties=font_prop)

# 前回との比較メッセージ
overall_improvement = df.iloc[-1]['accuracy_per'] - df.iloc[-2]['accuracy_per']
improvement_message = f"前回と比較して全体成績が{overall_improvement:.1f}ポイント"
improvement_message += "向上しました！" if overall_improvement > 0 else "低下しています。"
fig.text(0.5, 0.08, improvement_message, ha='center', fontsize=14, fontweight='bold',
         color='#2ecc71' if overall_improvement > 0 else '#e74c3c', fontproperties=font_prop)

# 今後の勉強アドバイス
weakest_area_idx = np.argmin(latest_values)
weakest_area = categories[weakest_area_idx]
advice = f"アドバイス: {weakest_area}の学習強化が最も効果的です。"
fig.text(0.5, 0.06, advice, ha='center', fontsize=13, fontweight='bold', color='#3498db', fontproperties=font_prop)

# 日数あたりの学習時間計算
# 前回と今回の学習期間を日数で計算
if len(df) >= 2:
    latest_date = df.iloc[-1]['date']
    previous_date = df.iloc[-2]['date']
    # timedelta.days + 1 で期間の日数を正確に計算（前後の日付を含まない）
    days_between = (latest_date - previous_date).days - 1
    
    latest_study_hours = df.iloc[-1]['study_hours']
    daily_study_minutes = (latest_study_hours * 60) / days_between
    
    # 目標: 1日あたり1.5時間 = 90分
    daily_target_minutes = 90
    
    # 1日あたりの学習時間メッセージ
    if daily_study_minutes < daily_target_minutes:
        time_status = "不足しています"
        color = '#e74c3c'  # 赤色（警告）
    else:
        time_status = "良好です"
        color = '#2ecc71'  # 緑色（良好）
    
    target_ratio = (daily_study_minutes / daily_target_minutes) * 100
    study_time_msg = f"1日あたりの学習時間: {daily_study_minutes:.1f}分 ({latest_study_hours:.1f}時間÷{days_between}日間) - 目標比:{target_ratio:.1f}% {time_status}"
else:
    study_time_msg = "学習データが不足しています"
    color = '#f39c12'  # オレンジ色

fig.text(0.5, 0.04, study_time_msg, ha='center', fontsize=14, fontweight='bold', color=color, fontproperties=font_prop)

# 全てのテキスト要素にフォントを適用
set_font_for_all_texts(fig)

# レイアウト調整とファイル保存
# 全体の余白を調整
plt.tight_layout(rect=[0.05, 0.12, 0.95, 0.96])

try:
    plt.savefig('score_report.png', dpi=300, bbox_inches='tight')
    print("[INFO] score_report.png を出力しました")
except Exception as e:
    print(f"[エラー] 画像保存中にエラーが発生しました: {e}")
