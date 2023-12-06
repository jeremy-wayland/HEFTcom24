# Energy Trading Approach

## Objective
For Trading track, we want to maximize revenue. The revenue formula is as follows:

`Revenue = Trade x DAP + (Actual - Trade) x (SSP - 0.07 x (Actual - Trade))` ,

where 0 <= Trade <= 1800 MWh (the maximum generation output of the hybrid power plant).

We need to find the value of Trade that maximizes this for every 30 minute slot.

## Background

**Trade** is the value we need to predict / optimize for. It is our market bid of how many kWh we will offer on the market for a particular slot (MW production x 0.5 for 30 mins slots).

**DAP** is day ahead price, in £/MWh, and should _hopefully_ be known by the time of predicting.

**Actual** is the actual generation. This is the same value as the target prediction in the energy forecasting track of the competition.

The difference between the volume of energy traded and actual generation is settled at the imbalance price (£/MWh). A participant’s imbalance price is given by `SSP - 0.07 x (Actual - Trade)`, where 0.07 is the regression coefficient between the net imbalance volume and imbalance price calculated from recent historic data, and therefore represents  the average impact  of  a  changes  in imbalance volume on the SSP.

**SSP** is calculated and distributed by Elexon. This price is subject to revision by Elexon and the most recent available data will be used in competition scoring.

## Approach

Essentially an optimization problem.
Assuming we have the accurate values of Actual, SSP, and DAP, then this task is essentially maximizing the following quadratic equation,
with the only constraints being the min and max values of Trade.

```
R = Revenue
D = DAP
A = Actual
S = SSP
x = Trade

R = D*x + (A - x)*(S - 0.07*(A - x))
```

Rearrange:
```
R = D*x + (A - x)*(S - 0.07A + 0.07x)
R = D*x + (A - x)*S - 0.07(A - x)(A - x)
R = D*x + A*S - S*x - 0.07(A² - 2Ax + x²)
R = -0.07x² + (D - S + 2A*0.07)x + (A*S - 0.07A²)
```
where 0 <= x <= 1800 MWh.

Since this is quatratic and concave down, the maximum of Revenue, R, should be at the x value where the derivative is 0.
```
0 = -0.14x + D - S + 2*0.07*A
0.14x = D - S + 0.14*A
x = ((S - D) / 0.14) + A
```

So, the maximum is not just simply making Trade = Actual. It depends on the other parameters too.
If we know all parameter values, the maximum is trivial.

**Therefore, the more accurate our values of SSP, DAP and the Actual production are, the more profitable our energy trade bid.**

The Forecasting track is already predicting Actual, and DAP should hopefully be given, so the next step would be to predict SSP.

For training we can use an Actual price that is either exactly accurate, or off by some random amount in either direction.
*Intuition:* doing the latter should help prevent overfitting because when we do the competition, we wont have the Actual production completely right,
since it will depend on our own power prediction. (We can test both approaches.)

SSP also seems to always be changing, according to the data in Energy_Data_20200920_20231027.csv. 

If DAP is not given, it will need to be predicted as well.

## Open Questions:
- Does SSP really have to be predicted or does Exelon give them beforehand fora period? What factors go into determining SSP?
- Will DAP for sure be known by the time of making a prediction? If not, will we need to predict that too / are there already good predictors out there?
- Is it naive to just assume the other variables are fixed to the correct value when doing a prediction for a particlar variable?
- Can we do joint predictions? (SSP and Actual together, for example)
- Do we even need time-based forecasting models for SSP? (Should plot prices over a certain amount of time to see if patterns).
    More importantly, is it itself time-dependent/cyclical in nature, or is it just based on the current values of other components
    that are cyclical in nature? If the later, traditional ML models with numerical prediction should probably do the trick.
