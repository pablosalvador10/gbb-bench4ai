from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()


def calculate_total_score(
    projected_acr: float,
    projected_length: float,
    partner_executives: float,
    actual_acr: float,
) -> float:
    """
    This function calculates a total score based on the projected Annual Contract Revenue (ACR),
    projected length of the project, number of partner executives involved in the project, and
    the actual Annual Contract Revenue (ACR).

    Each input is first normalized to a value between 0 and 1, then multiplied by a weight.
    The weights are constants specified on the whiteboard. The total score is the sum of these
    weighted values.

    Args:
    projected_acr (float): The projected Annual Contract Revenue (ACR). Expected to be between 0 and 50000.
    projected_length (float): The projected length of the project in months. Expected to be between 0 and 60.
    partner_executives (float): The number of partner executives involved in the project. Expected to be between 0 and 1.
    actual_acr (float): The actual Annual Contract Revenue (ACR). Expected to be between 0 and 100000.

    Returns:
    float: The total score, a weighted sum of the inputs. The score will be between 0 and the sum of the weights.
    """

    # Check the inputs and limit them if necessary
    if not 0 <= projected_acr <= 50000:
        logger.warning(
            f"Projected ACR {projected_acr} is not within the expected boundaries (0-50000). It will be limited."
        )
        projected_acr = max(min(projected_acr, 50000), 0)
        logger.warning(f"New Projected ACR: {projected_acr}")

    if not 0 <= projected_length <= 60:
        logger.warning(
            f"Projected length {projected_length} is not within the expected boundaries (0-60). It will be limited."
        )
        projected_length = max(min(projected_length, 60), 0)
        logger.warning(f"New Projected Length: {projected_length}")

    if not 0 <= actual_acr <= 100000:
        logger.warning(
            f"Actual ACR {actual_acr} is not within the expected boundaries (0-100000). It will be limited."
        )
        actual_acr = max(min(actual_acr, 100000), 0)
        logger.warning(f"New Actual ACR: {actual_acr}")

    # Constants from the whiteboard
    Wi = 0.4  # Weight for projected ACR
    Wq = 0.2  # Weight for projected length
    Wr = 0.2  # Weight for partner executives
    Wa = 0.2  # Weight for actual ACR

    # Normalizing the scores by the formula provided
    normalized_projected_acr = (projected_acr - 1000) / (50000 - 1000)
    normalized_projected_length = (projected_length - 1) / (60 - 1)
    normalized_partner_executives = (partner_executives - 0) / (1 - 0)
    normalized_actual_acr = (actual_acr - 0) / (100000 - 0)

    logger.debug(f"Normalized Projected ACR: {normalized_projected_acr}")
    logger.debug(f"Normalized Projected Length: {normalized_projected_length}")
    logger.debug(f"Normalized Partner Executives: {normalized_partner_executives}")
    logger.debug(f"Normalized Actual ACR: {normalized_actual_acr}")

    # Weighted values
    weighted_projected_acr = Wi * normalized_projected_acr
    weighted_projected_length = Wq * normalized_projected_length
    weighted_partner_executives = Wr * normalized_partner_executives
    weighted_actual_acr = Wa * normalized_actual_acr

    logger.debug(f"Weighted Projected ACR: {weighted_projected_acr}")
    logger.debug(f"Weighted Projected Length: {weighted_projected_length}")
    logger.debug(f"Weighted Partner Executives: {weighted_partner_executives}")
    logger.debug(f"Weighted Actual ACR: {weighted_actual_acr}")

    # Calculating total score based on the weights
    total_score = (
        weighted_projected_acr
        + weighted_projected_length
        + weighted_partner_executives
        + weighted_actual_acr
    )

    logger.debug(f"Total Score: {total_score}")

    return (
        total_score,
        weighted_projected_acr,
        weighted_projected_length,
        weighted_partner_executives,
        weighted_actual_acr,
    )
