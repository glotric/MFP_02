import numpy as np
import random
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation



def korak(x, y, mu):
    phi = random.uniform(0., 2*np.pi)
    a = 1.
    l = a * (random.uniform(0., 1.) ** (1 / (1-mu)))
    x += l * np.cos(phi)
    y += l * np.sin(phi)
    return x, y, l
    


def sprehod(dolzina, mu):
    #l_0 = v*t_0 = 1
    oddaljenost = 0
    r1 = np.zeros(dolzina+1)
    r2 = np.zeros(dolzina+1)
    x, y = 0., 0.
    ost = 0
    j = 0

    while oddaljenost < dolzina:
        xn, yn, l = korak(x, y, mu)
        ostn = (l + ost) % 1

        for i in range(int(l + ost)):
            if oddaljenost < dolzina:
                oddaljenost += 1
                r1[j+1] = x + ((xn - x) * (i+1 - ost) / l)
                r2[j+1] = y + ((yn - y) * (i+1 - ost) / l)
                j += 1
            else:
                return r1, r2
        x, y = xn, yn
        ost = ostn
        #print(oddaljenost)
    return r1, r2
                

def polet(dolzina, mu):
    #t = dolzina*t0, t0 = 1
    r1 = np.zeros(dolzina+1)
    r2 = np.zeros(dolzina+1)
    oddaljenost = np.zeros(dolzina+1)
    for i in range(dolzina):
        r1[i+1], r2[i+1], l = korak(r1[i], r2[i], mu)
        oddaljenost[i+1] = np.sqrt(r1[i+1] ** 2 + r2[i+1] ** 2)
    return r1, r2 #, oddaljenost


def vzporedno(funkcija, paralelno, dolzina, mu):
    logstep = [np.log(i) for i in range(2, dolzina+2)]
    logmad = np.zeros(dolzina)
    r1 = list()
    r2 = list()
    for i in range(paralelno):
        xi, yi = funkcija(dolzina, mu)
        r1.append(xi[1:])
        r2.append(yi[1:])

    for j in range(dolzina):
        logmad[j] = np.log(((median_abs_deviation([el[j] for el in r1]))**2 + (median_abs_deviation([el[j] for el in r2]))**2))

    return logmad, logstep


def linfunc(x, k, n):
    return k * np.array(x) + n



dolzina = 1000
paralelno = 1000


'''
mua = 1.5
xa = np.array(dolzina)
ya = np.array(dolzina)
xa, ya = polet(dolzina, mua)
mub = 2.5
xb = np.array(dolzina)
yb = np.array(dolzina)
xb, yb = polet(dolzina, mub)
muc = 5
xc = np.array(dolzina)
yc = np.array(dolzina)
xc, yc = polet(dolzina, muc)

plt.plot(xa, ya, label='$\mu=1.5$')
#plt.plot(xb, yb, label='$\mu=2.5$')
#plt.plot(xc, yc, label='$\mu=5$')
plt.title('Levyjev polet z 10000 koraki')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
'''
'''
logmad1 = np.zeros(dolzina)
logstep1 = np.zeros(dolzina)
mu1 = 1.5
logmad1, logstep1 = vzporedno(polet, paralelno, dolzina, mu1)
params1, err1 = curve_fit(linfunc, logstep1, logmad1)
k1 = params1[0].round(3)

plt.plot(logstep1, logmad1, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep1, linfunc(logstep1, params1[0], params1[1]), label=f'fit za $\mu=1.5$, $\gamma={k1}$', color='blue')

logmad2 = np.zeros(dolzina)
logstep2 = np.zeros(dolzina)
mu2 = 2.5
logmad2, logstep2 = vzporedno(polet, paralelno, dolzina, mu2)
params2, err2 = curve_fit(linfunc, logstep2, logmad2)
k2 = params2[0].round(3)

plt.plot(logstep2, logmad2, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep2, linfunc(logstep2, params2[0], params2[1]), label=f'fit za $\mu=2.5$, $\gamma={k2}$', color='orange')

logmad3 = np.zeros(dolzina)
logstep3 = np.zeros(dolzina)
mu3 = 5
logmad3, logstep3 = vzporedno(polet, paralelno, dolzina, mu3)
params3, err3 = curve_fit(linfunc, logstep3, logmad3)
k3 = params3[0].round(3)

plt.plot(logstep3, logmad3, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep3, linfunc(logstep3, params3[0], params3[1]), label=f'fit za $\mu=5$, $\gamma={k3}$', color='green')


plt.title('Poleti: 2log(MAD) v odvisnosti od logaritma časa za različne $\mu$')
plt.xlabel('log(t)')
plt.ylabel('2log(MAD)')
plt.legend()
plt.grid()
plt.show()
'''
'''
logmad1 = np.zeros(dolzina)
logstep1 = np.zeros(dolzina)
mu1 = 1.5
logmad1, logstep1 = vzporedno(sprehod, paralelno, dolzina, mu1)
params1, err1 = curve_fit(linfunc, logstep1, logmad1)
k1 = params1[0].round(3)

plt.plot(logstep1, logmad1, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep1, linfunc(logstep1, params1[0], params1[1]), label=f'fit za $\mu=1.5$, $\gamma={k1}$', color='blue')

logmad2 = np.zeros(dolzina)
logstep2 = np.zeros(dolzina)
mu2 = 2.5
logmad2, logstep2 = vzporedno(sprehod, paralelno, dolzina, mu2)
params2, err2 = curve_fit(linfunc, logstep2, logmad2)
k2 = params2[0].round(3)

plt.plot(logstep2, logmad2, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep2, linfunc(logstep2, params2[0], params2[1]), label=f'fit za $\mu=2.5$, $\gamma={k2}$', color='orange')

logmad3 = np.zeros(dolzina)
logstep3 = np.zeros(dolzina)
mu3 = 5
logmad3, logstep3 = vzporedno(sprehod, paralelno, dolzina, mu3)
params3, err3 = curve_fit(linfunc, logstep3, logmad3)
k3 = params3[0].round(3)

plt.plot(logstep3, logmad3, linestyle='', marker='.', markersize=2, color='gray')
plt.plot(logstep3, linfunc(logstep3, params3[0], params3[1]), label=f'fit za $\mu=5$, $\gamma={k3}$', color='green')


plt.title('Sprehodi: 2log(MAD) v odvisnosti od logaritma časa za različne $\mu$')
plt.xlabel('log(t)')
plt.ylabel('2log(MAD)')
plt.legend()
plt.grid()
plt.show()
'''

sp = 1.1
zg = 5
x = np.linspace(sp, zg, 1000)
ylet = np.ones(1000)
yhod = np.ones(1000)
for i in range(len(x)):
    if x[i] < 3:
        ylet[i] = 2/(x[i]-1)
for i in range(len(x)):
    if x[i] < 2:
        yhod[i] = 2
    elif 2 < x[i] < 3:
        yhod[i] = 4 - x[i]


muji = np.linspace(sp, zg, 50)
gamme = []
for mu in muji:
    logmad = np.zeros(dolzina)
    logstep = np.zeros(dolzina)
    logmad, logstep = vzporedno(polet, paralelno, dolzina, mu)
    params, err = curve_fit(linfunc, logstep, logmad)
    gamma = params[0].round(3)
    gamme.append(gamma)

plt.plot(muji, gamme, label='model', linestyle='', marker='.')
plt.plot(x, ylet, label='teorija')
plt.title('Odvisnost $\gamma$ od $\mu$ za polet')
plt.xlabel('$\mu$')
plt.ylabel('$\gamma$')
plt.legend()
plt.grid()
plt.show()