"""Main CLI entry point for MuSEEK pipeline execution.

This command runs MuSEEK pipelines defined in the museek.config module.
"""
import click
from ivory.cli.main import run as ivory_run


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        help_option_names=['-h', '--help'],
    )
)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def main(ctx, args):
    """Run MuSEEK data processing pipelines.
    
    \b
    USAGE:
      museek [OPTIONS] CONFIG_MODULE
    
    \b
    ARGUMENTS:
      CONFIG_MODULE    Pipeline configuration module (e.g., museek.config.process_uhf_band)
    
    \b
    OPTIONS:
      --PLUGIN-PARAM=VALUE    Override plugin parameters
                              Example: --InPlugin-block-name=1675632179
      -h, --help              Show this help message
    
    \b
    EXAMPLES:
      # Run demo pipeline
      museek museek.config.demo
      
      # Run UHF processing with custom parameters
      museek --InPlugin-block-name=1675632179 \\
             --InPlugin-context-folder=/custom/path \\
             museek.config.process_uhf_band
    
    \b
    AVAILABLE PIPELINES:
      museek.config.demo                    - Demo pipeline
      museek.config.process_uhf_band        - UHF band data processing
      museek.config.process_l_band          - L band data processing
      museek.config.sanity_check            - Observation sanity checks
    
    \b
    SLURM JOB SUBMISSION:
      For SLURM job submission with common configurations, use:
        museek_run_process_uhf_band --help
        museek_run_notebook --help
    
    For more information, see the MuSEEK documentation.
    """
    # If no arguments provided, show help
    if not args:
        click.echo(ctx.get_help())
        ctx.exit(0)
    
    # Pass all arguments through to ivory's run function
    import sys
    sys.argv = ['museek'] + list(args)
    ivory_run()


if __name__ == "__main__":
    main()
