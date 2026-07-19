import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def create_architecture(output=None):

    # Create output folder automatically
    if output is None:
        output = Path("figures") / "11_system_architecture.pdf"
    else:
        output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fig, ax = plt.subplots(
        figsize=(12,5)
    )

    ax.axis("off")


    blocks = [
        ("DP-16QAM\nTransmitter",0.05),
        ("S+C+L\nWDM Comb",0.20),
        ("Fiber Link\nISRS+Kerr+CD",0.35),
        ("Raman + EDFA\nAmplification",0.50),
        ("Coherent\nReceiver DSP",0.65),
        ("GSNR/EVM/GMI\nMetrics",0.80)
    ]


    for text,x in blocks:

        ax.add_patch(
            Rectangle(
                (x,0.4),
                0.12,
                0.25
            )
        )

        ax.text(
            x+0.06,
            0.525,
            text,
            ha="center",
            va="center",
            fontsize=10
        )


    for i in range(len(blocks)-1):

        ax.annotate(
            "",
            xy=(blocks[i+1][1],0.525),
            xytext=(blocks[i][1]+0.12,0.525),
            arrowprops={
                "arrowstyle":"->"
            }
        )


    plt.savefig(
        str(output),
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    create_architecture()
    print(
        "System architecture figure generated successfully"
    )