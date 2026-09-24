"""Release each Cube renderer before another OSMesa environment is created."""


def install_cube_renderer_cleanup(ew):
    """Cover the frozen shared-cache factory as well as every new rollout.

    The installed Cube class inherits a no-op close; its renderer otherwise
    survives until garbage collection and can destroy a newer OSMesa context.
    This process-local binding changes resource release only.
    """
    original_make_live = ew.runtime._make_live

    def make_live(context, row):
        values = original_make_live(context, row)
        env = values[0]
        base = env.unwrapped
        original_close = base.close

        def close():
            renderer = base.__dict__.get('_renderer')
            if renderer is not None:
                renderer.close()
                base._renderer = None
            return original_close()

        base.close = close
        return values

    ew.runtime._make_live = make_live
