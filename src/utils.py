def build_top_factors(features: dict) -> list[str]:
    factors = []

    for name, label in [
        ("ROUTE_RISK", "Route historically has higher delay likelihood"),
        ("UNIQUE_CARRIER_RISK", "Carrier has higher historical delay likelihood"),
        ("ORIGIN_RISK", "Origin airport is historically delay-prone"),
        ("DEST_RISK", "Destination airport is historically delay-prone"),
    ]:
        v = features.get(name)
        if v is not None and float(v) > 0.18:
            factors.append(label)

    if features.get("DEP_HOUR_BIN") in ["15-17", "18-20", "21-23"]:
        factors.append("Departure time falls in a higher congestion window")

    if features.get("SEASON") == "Summer":
        factors.append("Summer travel periods tend to have higher congestion-driven delays")

    return factors[:4] if factors else ["Delay risk is driven by combined route, carrier, and congestion patterns"]
