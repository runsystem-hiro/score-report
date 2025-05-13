# 試験成績レポート自動生成ツール

このプロジェクトは、CSV形式で蓄積された学習・試験データをもとに、視覚的なレポートを自動生成するPythonツールです。全体の正解率、各分野別の成績推移、学習進捗、自己学習時間などをグラフィカルに可視化し、学習状況を一目で把握できるレポートを出力します。

## 📊 出力例

![score_report](score_report.png)

## 🧩 特徴

- **全体正解率の時系列推移グラフ**
- **分野別成績（ストラテジ系・マネジメント系・テクノロジ系）のレーダーチャート**
- **各分野の改善度の棒グラフ**
- **分野別の学習進捗率の折れ線グラフ**
- **自己学習時間の推移（棒グラフ）**
- **評価を含む成績サマリーテーブル**
- **自動的な改善メッセージ表示（前回との比較）**

## 📁 ファイル構成

```bash
.
├── plot_score_report.py        # レポート生成用スクリプト
├── score_report_final.csv      # 入力CSV（学習・成績データ）
├── score_report.png            # 出力されるレポート画像
└── README.md                   # 本ドキュメント
```

## 🔧 必要環境

- Python 3.8 以上
- 以下の主要ライブラリ：

```bash
pip install matplotlib pandas numpy
```

## 📝 入力CSVフォーマット

`score_report_final.csv` には以下の列が含まれている必要があります（順不同）：

| 列名 | 説明 |
|------|------|
| `date` | 報告日（YYYY-MM-DD） |
| `accuracy_per` | 正解率 (%) |
| `total_questions` | 問題数 |
| `correct_answers` | 正解数 |
| `grade` | 評価（例: C-, B+） |
| `strategy_per`, `management_per`, `technology_per` | 各分野の成績 (%) |
| `strategy_progress`, `management_progress`, `technology_progress` | 各分野の学習進捗 (%) |
| `study_hours` | 自己学習時間（単位: 時間） |

例：

```csv
date,accuracy_per,total_questions,correct_answers,grade,strategy_per,management_per,technology_per,strategy_progress,management_progress,technology_progress,study_hours
2025-04-15,0,0,0,D,0,0,0,0,0,0,0
2025-04-30,41.0,100,41,C-,31.3,43.5,46.7,20,40,40,2
...
```

## 🚀 実行方法

```bash
python plot_score_report.py
```

実行すると、同フォルダ内に `score_report.png` が生成されます。

## 🗒 補足

- フォント設定は日本語対応の `BIZ UDGothic` に最適化されています。システムにインストールされていない場合は、適宜変更してください。
- グラフ出力は `plt.savefig(...)` によってPNGファイルに保存されます。GUI環境がないサーバーでも問題なく動作します。
