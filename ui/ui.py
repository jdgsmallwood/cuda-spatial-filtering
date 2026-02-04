from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Select,
    Button,
    Log,
    Input,
    Label,
    Collapsible,
)
from textual.containers import Vertical, Grid
from textual.validation import Integer
from textual import work
import asyncio
import os


class CMakeBuilder(App):
    CSS = """
    Grid { grid-size: 2; grid-columns: 1fr 1fr; height: auto; padding: 1; }
    .input-group { margin: 0 1 1 1; }
    Input.-valid { border: tall $success; }
    Input.-invalid { border: tall $error; }
    Log { background: $panel; border: solid $accent; height: 1fr; margin-top: 1;}
    #build_btn { width: 100%; margin: 1 0; }
    """

    CONFIG_SCHEMA = {
        "nr_observing_buffers": ("3", Integer(minimum=1)),
        "nr_observing_fpga_sources": ("1", Integer(minimum=1)),
        "nr_observing_channels": ("8", Integer(minimum=1)),
        "nr_observing_receivers_per_packet": ("10", Integer(minimum=1)),
        "nr_observing_packets_for_correlation": ("256", Integer(minimum=10)),
        "nr_observing_correlation_blocks_to_integrate": ("56", Integer(minimum=10)),
    }

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Collapsible(title="Observing Parameters", collapsed=False):
                with Grid():
                    for key, (default, validator) in self.CONFIG_SCHEMA.items():
                        with Vertical(classes="input-group"):
                            yield Label(f"{key}:")
                            yield Input(value=default, validators=[validator], id=key)

            yield Select(
                [("Debug", "Debug"), ("Release", "Release")],
                value="Release",
                id="config_type",
            )
            yield Button("Configure & Build", variant="success", id="build_btn")
            yield Log(id="build_log")
        yield Footer()

    def round_up_to_32(n):
        # (n + mask) & ~mask
        return (n + 31) & ~31

    def on_input_changed(self) -> None:
        all_valid = all(inp.is_valid for inp in self.query(Input))
        self.query_one("#build_btn", Button).disabled = not all_valid

    @work(exclusive=True, thread=True)
    async def run_build(self, flags: list):
        btn = self.query_one("#build_btn", Button)
        btn.disabled = True  # Lock the button during build
        log = self.query_one(Log)
        log.clear()
        log.write("Starting conda build...\n")

        env_name = "base"

        cmds = [
            ["conda", "run", "-n", env_name, "cmake", "."] + flags,
            ["conda", "run", "-n", env_name, "cmake", "--build", "."],
        ]

        for cmd in cmds:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                log.write(line.decode())
            await proc.wait()
            if proc.returncode != 0:
                log.write("\nBUILD FAILED\n")
                btn.disabled = False
                return

        btn.disabled = False
        log.write(f"\nSUCCESS! Binary located in build folder.\n")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "build_btn":
            flags = [
                f"-D{k.upper()}={self.query_one(f'#{k}', Input).value}"
                for k in self.CONFIG_SCHEMA.keys()
            ]
            flags.append(
                f"""-DNR_OBSERVING_PADDED_RECEIVERS={
                    self.round_up_to_32(
                        int(
                            self.query_one(f"#{nr_observing_fpga_sources}", Input).value
                        )
                        * int(
                            self.query_one(
                                f"#{nr_observing_receivers_per_packet}", Input
                            ).value
                        )
                    )
                }"""
            )
            flags.append(
                f"-DCMAKE_BUILD_TYPE={self.query_one('#config_type', Select).value}"
            )
            self.run_build(flags)


if __name__ == "__main__":
    CMakeBuilder().run()
