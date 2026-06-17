import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Ajuste a URI se estiver usando o rádio Crazyradio (ex: 'radio://0/80/2M')
# Se estiver no cabo USB, geralmente é 'usb://0'
URI = 'radio://0/80/2M/E7E7E7E7E7'

def main():
    # 1. "Header" de inicialização obrigatório da biblioteca
    cflib.crtp.init_drivers()

    print(f"Conectando ao Crazyflie em {URI}...")
    
    # 2. Abre a conexão
    with SyncCrazyflie(URI) as scf:
        print("Conectado! Extraindo o Índice (TOC) de parâmetros...\n")
        
        # 3. Imprime tudo
        print("--- Lista de Parâmetros Disponíveis ---")
        for group in scf.cf.param.toc.toc.keys():
            for name in scf.cf.param.toc.toc[group].keys():
                print(f"{group}.{name}")
        print("---------------------------------------")

if __name__ == '__main__':
    main()