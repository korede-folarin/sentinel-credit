"""
Sweeps the miss hazard and cure generosity to find a
combination producing plausible delinquency and default rates.

Narrowed around the region identified by the wide sweep:
halflife 45 days, hazard near 0.004, cure scale between
1.0 and 1.5.
"""
from bnpl_data_generator import (
    generate_customers,
    generate_transactions,
    calibrate_grid,
)

if __name__ == "__main__":
    print("Generating population for calibration...")
    customers = generate_customers()
    transactions = generate_transactions(customers)
    calibrate_grid(
        customers, transactions,
        hazards=[0.004, 0.005, 0.006],
        cure_scales=[1.1, 1.2, 1.3, 1.4, 1.5],
        halflives=[45],
    )