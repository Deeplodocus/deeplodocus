from deeplodocus.core.project.structure.transformers import \
    DEEP_CONFIG_COMMENT, \
    DEEP_CONFIG_DEFAULT, \
    DEEP_CONFIG_INIT, \
    DEEP_CONFIG_WILDCARD


def generate_transformer(transformer, filename):
    lines = __generate_lines(transformer)
    with open(filename, "w") as file:
        file.writelines("\n".join(lines))


def __generate_lines(transformer, tab_size=2, tabs=0, in_list=False):
    pre = "- " if in_list else ""
    lines = []
    for key, items in transformer.items():
        if key.startswith(DEEP_CONFIG_COMMENT):
            lines.append(items)
        # If items are a dictionary
        if isinstance(items, dict):
            # If the key is a wildcard, and it has an init value, print the init value as the key
            if key == DEEP_CONFIG_WILDCARD and DEEP_CONFIG_INIT in items:
                key = items[DEEP_CONFIG_INIT]
                lines.append("%s%s%s:" % (" " * tab_size * tabs, pre, key))
            # If items include a default or an init value, print the key and value
            elif DEEP_CONFIG_DEFAULT in items or DEEP_CONFIG_INIT in items:
                value = items[DEEP_CONFIG_INIT] if DEEP_CONFIG_INIT in items else items[DEEP_CONFIG_DEFAULT]
                lines.append("%s%s%s: %s" % (" " * tab_size * tabs, pre, key, value))
            # Else, just print the given key
            else:
                lines.append("%s%s%s:" % (" " * tab_size * tabs, pre, key))
            lines += __generate_lines(items, tabs=tabs+1)
        # If items are a list (wildcards can be a list)
        elif isinstance(items, list):
            lines.append("%s%s%s:" % (" " * tab_size * tabs, pre,  key))
            for item in items:
                lines += __generate_lines(item, tabs=tabs+1, in_list=True)
    return lines


