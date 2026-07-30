"""Checks where in the schedule defaults originate."""
import pandas as pd
from bnpl_data_generator import diagnose_default_timing

repayments = pd.read_csv("data_generator/output/repayments.csv")
diagnose_default_timing(repayments)