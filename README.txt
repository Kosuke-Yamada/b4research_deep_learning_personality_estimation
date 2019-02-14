

model1
単語のBag-of-Wordsを用いて，全結合モデルで性格推定．

model2
単語のBag-of-Wordsを単語数に変更したものを用いて，全結合モデルで性格推定．

model3
ツイートをWord-embedding(同時に学習)にし，全結合モデルで性格推定．

model4
ツイートをWord-embedding(同時に学習)にし，Bi-LSTMモデルで性格推定．

model5
性格指標4つを同時に学習させる．
ツイートをWord-embedding(同時に学習)にし，全結合モデルで性格推定．
