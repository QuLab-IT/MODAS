import numpy as np

def detect_entropy_anomalies(Sxx_anomalies, t_anomalies, M_test, upper_percentile=95, lower_percentile=5):
    # Compute entropy across frequencies
    entropy = -np.sum(Sxx_anomalies * np.log(Sxx_anomalies + 1e-10), axis=0)

    # Thresholds based on percentiles
    upper_threshold = np.percentile(entropy, upper_percentile)
    lower_threshold = np.percentile(entropy, lower_percentile)

    # Detect interpolated threshold crossings
    crossings = []
    for condition in [entropy > upper_threshold, entropy < lower_threshold]:
        prev = condition[0]
        start_time = None
        for i in range(1, len(condition)):
            curr = condition[i]
            if not prev and curr:
                t1, t2 = t_anomalies[i - 1], t_anomalies[i]
                e1, e2 = entropy[i - 1], entropy[i]
                th = upper_threshold if condition is (entropy > upper_threshold) else lower_threshold
                t_cross_start = t1 + (th - e1) / (e2 - e1) * (t2 - t1)
                start_time = t_cross_start
            elif prev and not curr and start_time is not None:
                t1, t2 = t_anomalies[i - 1], t_anomalies[i]
                e1, e2 = entropy[i - 1], entropy[i]
                th = upper_threshold if condition is (entropy > upper_threshold) else lower_threshold
                t_cross_end = t1 + (th - e1) / (e2 - e1) * (t2 - t1)
                crossings.append((start_time, t_cross_end))
                start_time = None
            prev = curr

    # Create anomaly mask based on crossings
    anomaly_mask = np.zeros_like(M_test, dtype=bool)
    for t_start, t_end in crossings:
        t_start_M = t_start / 60  # Convert seconds to minutes
        t_end_M = t_end / 60
        mask = (M_test >= t_start_M) & (M_test <= t_end_M)
        anomaly_mask |= mask

    anomaly_indices = np.where(anomaly_mask)[0]

    return anomaly_indices, anomaly_mask, crossings