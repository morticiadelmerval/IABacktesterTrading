import backtester

def test_run():
    print("Corriendo con 0% commission...")
    backtester.run_all(commission=0.0)

if __name__ == "__main__":
    test_run()
