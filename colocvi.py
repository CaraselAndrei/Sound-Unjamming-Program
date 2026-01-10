import numpy as np               # Pentru lucrul cu matrici și vectori
import matplotlib.pyplot as plt  # Pentru grafice (vizualizare)
from scipy.io import wavfile     # Pentru a citi/scrie fișiere .wav
import scipy.linalg as la        # Pentru funcții standard de comparatie (ex: qr, solve)


def desc_qr(A):
    #Calculează descompunerea QR a matricei A folosind Gram-Schmidt Modificat.
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Copiem coloanele lui A în Q pentru procesare
    for j in range(n):
        v = A[:, j]
        
        # O ortogonalizăm față de coloanele anterioare
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        
        # Calculăm norma vectorului rămas
        R[j, j] = np.linalg.norm(v)
        
        # Normalizăm coloana și o punem în Q
        Q[:, j] = v / R[j, j]
        
    return Q, R

def rez_qr(A, b):
    """ Rezolvă Ax = b folosind QR scris de noi """
    # 1. Descompunem A
    Q, R = desc_qr(A)
    
    # 2. Calculăm y = Q_transpus * b
    y = np.dot(Q.T, b)
    
    # 3. Rezolvăm sistemul triunghiular R * x = y (prin substituție inversă)
    # Putem folosi o funcție simplă pentru asta sau solve din scipy pentru triunghiular
    x = np.linalg.solve(R, y) 
    
    return x

fs, data = wavfile.read('Sunete\HarryPotter.wav')

print(f"Frecvența de eșantionare: {fs}")
print(f"Tipul datelor originale: {data.dtype}")

if len(data.shape) > 1:
    data = data.mean(axis=1)
    print("Am convertit sunetul din Stereo in Mono.")


limita_timp=int(input("Introduce-ti limita de timp(numar natural): "))
if len(data) > fs * limita_timp:
    data = data[:fs * limita_timp]

data = data / np.max(np.abs(data))

time_axis = np.linspace(0, len(data) / fs, num=len(data))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, data, label='Semnal Original')
plt.title("Semnalul Audio Inițial")
plt.xlabel("Timp (secunde)")
plt.ylabel("Amplitudine")
plt.legend()
plt.grid(True)
plt.show()
plt.pause(0.001)

intensitate=float(input("Introduce-ti intensitatea zgomotului de fundal (intre 0.0 si 1.0): "))
print("Se bruiaza sunetul...")
zgomot = np.random.normal(0, intensitate, data.shape)

print("Încep procesarea (poate dura puțin)...")
data_zgomotos = data + zgomot
# Parametrii algoritmului
fereastra = 15       # Câte puncte luăm odata (ex: 15 puncte)
grad_polinom = 2    # Gradul polinomului de aproximare (parabolă)
n = len(data_zgomotos)
data_filtrata = np.zeros(n)

# Construim matricea A o singură dată (este aceeași pentru orice fereastră!)
# A este matrice Vandermonde pe axa [-k, ..., 0, ..., k]
half_win = fereastra // 2
x_local = np.linspace(-1, 1, fereastra) # Axa x normalizată local
A = np.vander(x_local, grad_polinom + 1) # Matricea sistemului

for i in range(half_win, n - half_win):
    # 1. Extragem bucata curentă de sunet (vectorul b local)
    b_local = data_zgomotos[i - half_win : i + half_win + 1]
    
    # 2. Rezolvăm sistemul A * x = b_local folosind QR-ul nostru
    coeficienti = rez_qr(A, b_local)
    
    # 3. Valoarea filtrată este valoarea polinomului în centru
    # Polinomul este c0*x^2 + c1*x + c2.
    # În centrul ferestrei, coordonata x_local este aprox 0.
    # Putem recalcula tot vectorul estimat:
    b_estimat = np.dot(A, coeficienti)
    
    # Luăm doar punctul din mijloc
    data_filtrata[i] = b_estimat[len(b_estimat)//2]

print("Procesare finalizată!")

plt.figure(figsize=(12, 6))
plt.plot(data_zgomotos, color='lightgray', label='Semnal cu Zgomot')
plt.plot(data, color='green', alpha=0.6, label='Original')
plt.plot(data_filtrata, color='red', linestyle='--', label='Filtrat (Algoritmul tau)')
plt.xlim(1000, 1200) # Facem zoom pe o zonă mică să se vadă diferența
plt.legend()
plt.title("Rezultatul Netezirii folosind CMMP și QR")
plt.show()
plt.pause(0.001)

wavfile.write('rezultat_final.wav', fs, (data_filtrata * 32767).astype(np.int16))
nume_fisier_zgomot = 'sunet_cu_zgomot.wav'
wavfile.write(nume_fisier_zgomot, fs, (data_zgomotos * 32767).astype(np.int16))
print(f"Am salvat {nume_fisier_zgomot}")