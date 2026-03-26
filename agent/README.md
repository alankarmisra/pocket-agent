# Python Source Files Overview

## main.py

- `main()`: Entry point of the application that orchestrates program execution.
- `load_config(path)`: Loads configuration settings from a JSON file.
- `initialize_logger()`: Sets up and configures the application logger.

## utils.py

- `format_timestamp(dt)`: Converts a datetime object to a standardized ISO format string.
- `validate_email(email)`: Checks if an email address conforms to standard email formatting rules.
- `safe_divide(a, b)`: Performs division safely, returning None instead of raising ZeroDivisionError.

## models.py

- `User(name, email)`: Represents a user entity with name and email attributes.
- `load_user_from_dict(data)`: Creates a User instance from a dictionary of attributes.
- `save_user_to_db(user)`: Saves a User object to the database and returns the database ID.

## config.py

- `get_db_url()`: Constructs and returns the database connection URL from environment variables.
- `is_debug_mode()`: Returns a boolean indicating whether debug mode is enabled.
