is_simple_core = False

if is_simple_core:
    from kdnn.core_simplified import Variable
    from kdnn.core_simplified import Function
    from kdnn.core_simplified import using_config
    from kdnn.core_simplified import no_grad
    from kdnn.core_simplified import as_array
    from kdnn.core_simplified import as_variable
    from kdnn.core_simplified import setup_variable
else:
    from kdnn.core import Variable
    from kdnn.core import Function
    from kdnn.core import using_config
    from kdnn.core import no_grad
    from kdnn.core import as_array
    from kdnn.core import as_variable
    from kdnn.core import setup_variable

setup_variable()
