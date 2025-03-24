### Metrics between` y`and `y_pred`

#### 1.Cosine Similarity

$$ \operatorname{cosine} \_\operatorname{sim}(A, B)=\frac{A \cdot B}{\|A\|\|B\|} $$

#### 2.Correlation

$$ \operatorname{corr}(X, Y)=\frac{\sum\left(X_i-\bar{X}\right)\left(Y_i-\bar{Y}\right)}{\sqrt{\sum\left(X_i-\bar{X}\right)^2} \sqrt{\sum\left(Y_i-\bar{Y}\right)^2}} $$

#### 3.binary precision

```
 np.sum((y_pred * y_true > 0))/ len(y_pred) 
```

正负号符号预测一致的比例

#### 4. reward

$$ \frac{\sum_{i=1}^n\left(y_i-\bar{y}\right)\left(y_{\text {pred }, i}-\bar{y}_{\text {pred }}\right)}{\sqrt{\sum_{i=1}^n\left(y_{\text {pred }, i}-\bar{y}_{\text {pred }}\right)^2}} $$ 

这个指标适用于需要衡量预测值与真实值线性关系的场景，但更关注预测值变化对真实值变化的解释能力

### Metrics for `y_pred`

#### 1. Mean 

$ \mu = \frac{1}{n} \sum_{i=1}^{n} y_i $

#### 2. Variance 

$$ \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mu)^2 $$

#### 3. Standard Deviation/Volatility

$$ \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \mu)^2} $$

用于了解预测的波动性和变化程度

#### 4. Mean Divided by Standard Deviation 

$ \text{Ratio} = \frac{\mu}{\sigma} $

我们希望mean/variance是尽量接近0的,也就是不希望预测成一个有bias的信号

#### 5. skewness

$$ S = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{y_i - \mu}{\sigma}\right)^3 $$

预测值分布的不对称程度,正偏度表示右尾较长的分布，而负偏度表示左尾较长的分布

#### 6.kurtosis

$$ K = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left(\frac{y_i - \mu}{\sigma}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)} $$

预测值分布的"尖峰度",较高的峰度表示分布具有较重的尾部（更多极端值），较低的峰度表示尾部较轻。