import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("..\Kursmaterialien\Abschnitt 19 - Projekt Statistische Grundlagen\sf_salaries.csv",
    low_memory = False)

#print(df.head())

# Vergeleich Salary der Jahre 2013, 2012, 2011

df_2013 = df[df["Year"] == 2013]
#print(df_2013.head())
mean_2013 = df_2013["TotalPay"].mean()
print("mean salary 2013 (in $): " + str(mean_2013))

df_2012 = df[df["Year"] == 2012]
mean_2012 = df_2012["TotalPay"].mean()
print("mean salary 2012 (in $): " + str(mean_2012))

df_2011 = df[df["Year"] == 2011]
mean_2011 = df_2011["TotalPay"].mean()
print("mean salary 2011 (in $): " + str(mean_2011))

mean_total = df["TotalPay"].mean()
print("mean salary total (in $): " + str(mean_total))

median_total = df["TotalPay"].median()
print("median salary total (in $): " + str(median_total))

plt.hist(df["TotalPay"], bins = 1000)
plt.xlabel("amount")
plt.ylabel("Salary (in $)")
plt.show()