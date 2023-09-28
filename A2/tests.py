'''
   Testing to determine the best parameters to use for our decision tree.
   The factors we want to optimize for are
       1) Max Depth
       2) MinLeafSize
       3) MinSampleSplit

results= []


X_Train = [0]*10
Y_Train = [0]*10
X_Test = [0]*10
Y_Test = [0]*10

for i in range(0,10):

    train, test = train_test_split(data, test_size=0.1)

    Y_Train[i] = train.iloc[:, 0]
    X_Train[i] = train.iloc[:, 1:]

    Y_Test[i] = test.iloc[:, 0]
    X_Test[i] = test.iloc[:, 1:]
regr_1 = DecisionTreeRegressor()
for maxDepth in range(5,7):
    for minLeafSize in range(40,70):
        for minSplitSize in range(3,20):
            regr_1.set_params(**{"max_depth": maxDepth, "min_samples_leaf": minLeafSize, "min_samples_split": minSplitSize})

            mse = 0
            for i in range(0,10):
                regr_1.fit(X_Train[i], Y_Train[i])
                vals = regr_1.predict(X_Test[i])
                for x in range(0, len(vals)):
                    mse += ((vals[x] - Y_Test[i].iloc[x])**2)

            # Calculate Error in Samples and report it back

            results.append((int(mse), str(maxDepth) + " | " + str(minLeafSize) + " | " + str(minSplitSize)))


results.sort(key=lambda a: a[0], reverse=True)

x = True
first = 0
while x:
    val = results.pop()
    if (first == 0):
        first = val[0]
    if (val[0] != first):
        x= False
    print(val)

for x in range(0,20):
    print(results.pop())

====== RESULTS ====
Best Depth: 6
Best Min Leaf Size: 50
Best Split: 19
'''