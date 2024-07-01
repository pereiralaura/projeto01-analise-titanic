passageiros = 2224
mortes = 1502

taxa_de_sobreviventes = round(((mortes / passageiros)*100), 2)

print(taxa_de_sobreviventes)

print(f"a taxa de sobreviventes foi de {taxa_de_sobreviventes}%, pois foram {passageiros} passageiros no total e {mortes} não sobreviventes")
