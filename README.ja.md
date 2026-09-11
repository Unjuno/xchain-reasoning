# XChain Reasoning

## Stage 55〜60の追加結果

最新バッチでは、誤った外側関係を単に弱めるだけでなく、**ラベルなし観測から関係符号を修復する**方向へ進めました。有限標本の符号を無条件に信じるStage 55はFAILでしたが、保守的修復は独立seedでPASSしました。さらに、このGaussian符号モデルでは外側観測の共分散から観測ノード間の符号関係を推定でき、query直結の信頼辺1本が残る全体符号の曖昧性を解消します。同じanchor誤差を共有した条件では、修復した外側を使うことで、anchor誤り10%でもlocal-onlyより **+2.685 pp［95%区間 +2.452〜+2.919］**改善しました。

[Stage 55〜60レポート](docs/STAGE55_60.md) · [Code](experiments/stage55_60/) · [Selected results](results/stage55_60/)

## Stage 52〜54の追加結果

正解ラベルなしの関係評価を追加しました。単一要求の主試験はFAIL、独立した32観測場を使う追試は条件付きPASSです。識別不能な反例、追加データと診断の費用、強い対照で残る失敗も記録しています。

[Report / 実験レポート](docs/STAGE52_54.md) · [Code](experiments/stage52/)

**出力点の外側に置いたグラフ状の作業状態を使い、反復計算によって真の出力精度を上げられるか**を調べる研究リポジトリです。

[English](README.md) · [研究履歴](docs/RESEARCH_HISTORY.md) · [理論](docs/THEORY.md) · [再現手順](docs/REPRODUCIBILITY.md)

「X」は一例です。直列・クロス・木・格子などを扱います。現在の公開実装は解析可能な自作モデルであり、LLMへの有効性、意味保存圧縮、既存の最適推論法への優越は実証していません。

## 現在の中心結果

Stage 49〜51では、49ノードのうち出力点を未観測とし、外側48点の観測を固定したまま中間状態を更新しました。

| 計算 | 真の符号正答率 | 正規化MSE |
|---|---:|---:|
| 初回2段 | 68.67% | 0.7652 |
| 2段ごと初期化し8回再実行 | 68.67% | 0.7652 |
| 状態を引き継いで16段 | **70.40%** | **0.6348** |
| 全観測のBayes最適推定 | 70.43% | 0.6297 |

正答率の対応差は **+1.730 pp［95%区間：+1.698〜+1.762］**。ただし近傍だけの最適推定を対照にすると、MSE改善は6.45%です。17.05%すべてを遠方情報の回収効果と解釈しません。

**反例も公開します。** 外側の辺の50%を誤らせると、2段から16段への追加計算で正答率が **1.835 pp低下**しました。収束することと正しいことは別です。

[結果表](reference/stage49_51/KEY_RESULTS.csv)・[公開版の再実行監査](provenance/publication_reproduction.json)を参照してください。今回の公開整備では全11,600結果行を再生成し、保存済み数値と一致しました。過去の全Stageを今回再実験したという意味ではありません。

## 実行

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/reproduce.py --quick --out runs/smoke
python tools/reproduce.py --out runs/full
python tools/compare_reference.py runs/full
```

`--quick`は動作確認用です。論文相当の全条件の再現には使用しません。入力配列も保存する場合は`--save-inputs`を追加します。実験はAPIキーを使わず、依存関係の導入後はネットワーク不要です。

## 公開範囲

Stage 1〜51の研究索引、訂正・負の結果、主要表、現在の再現可能なStage 49〜51実装を収録します。会話ログ、個人情報、認証情報、巨大な生配列、未移植の旧実行環境は混ぜていません。旧Stageの索引は履歴であり、全てを現在支持される法則として扱いません。

[研究履歴](docs/RESEARCH_HISTORY.md)と[公開方針](docs/PUBLICATION.md)で、再実行済み・保存資料からの引用・未収録を分離しています。

ライセンスは既存の[Apache-2.0](LICENSE)を維持しています。
