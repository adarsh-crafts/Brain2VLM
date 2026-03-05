import subprocess
import re

methods = ["cvpr", "mlp"]
features = [
    "alexnet5",
    "alexnet12",
    "alexnet18",
    "clip_h6",
    "clip_h12",
    "clip",
    "inception"
]

subject = "subj01"

results = {}

for method in methods:
    results[method] = {}

    for feat in features:
        cmd = [
            "python",
            "identification.py",
            "--usefeat", feat,
            "--subject", subject,
            "--method", method
        ]

        print(f"Running {method} - {feat}")

        output = subprocess.check_output(cmd, text=True)

        # extract ACC value
        match = re.search(r"ACC = ([0-9.]+)", output)
        if match:
            acc = float(match.group(1))
            results[method][feat] = acc
        else:
            results[method][feat] = None


# Write results
with open("identification_comparison.txt", "w") as f:

    header = "Feature\t" + "\t".join(methods) + "\n"
    f.write(header)

    for feat in features:
        line = [feat]
        for method in methods:
            val = results[method].get(feat, None)
            if val is None:
                line.append("NA")
            else:
                line.append(f"{val:.3f}")

        f.write("\t".join(line) + "\n")

print("Saved results to identification_comparison.txt")