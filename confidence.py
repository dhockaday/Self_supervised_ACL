import scipy.stats
import numpy as np
confidence = 0.95  # Change to your desired confidence level
z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
print(z_value)


acc_test = 0.337
sample = 947

ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / sample)

ci_lower = acc_test - ci_length
ci_upper = acc_test + ci_length

print(ci_lower, ci_upper)
