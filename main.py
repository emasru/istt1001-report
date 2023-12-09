import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("lego.population.csv", sep = ",", encoding = "latin1")

# Fjerner forklaringsvariabler vi ikke trenger
df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages', 'Minifigures', 'Unique_Pieces']]

# Fjerner observasjoner med manglende datapunkter
df2 = df2.dropna()

# Gjør themes om til string og fjern alle tegn vi ikke vil ha med
df2['Theme'] = df2['Theme'].astype(str)
df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex = True)

# Fjerner dollartegn og trademark-tegn fra datasettet
df2['Price'] = df2['Price'].str.replace('\$', '', regex = True)

# Og gjør så prisen om til float
df2['Price'] = df2['Price'].astype(float)

# Gruppere temaer i nye grupper:
# (Harry Potter, NINJAGO og Star Wars havner i én gruppe, City og Friends i en annen, og alle andre i en tredje)

df2['Gender'] = np.where(df2['Theme'].isin(["NINJAGO", "Star Wars", "Spider-Man", "Batman",
"Marvel", "DC", "Hidden Side", "Speed Champions", "Monkie Kid", "City", "Jurassic World"]), 'gutt',
 np.where(df2['Theme'].isin(["Friends", "Unikitty", "LEGO Frozen 2", "Powerpuff Girls", "Trolls
World Tour"]), 'jente', 'nøytral'))
df2.groupby(['Gender']).size().reset_index(name = 'Count')

# Regresjon
A = 'Price ~ Pieces'
B = 'Price ~ Pieces + Pages'
C1 = 'Price ~ Pieces + Gender'
C2 = 'Price ~ Pieces*Gender'

formel = C2
modell = smf.ols(formel, data = df2)
resultat = modell.fit()
resultat.summary()

# Plotte predikert verdi mot residual
figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = modell.fit().fittedvalues, y = modell.fit().resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

# Lage kvantil-kvantil-plott for residualene
sm.qqplot(modell.fit().resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.show()