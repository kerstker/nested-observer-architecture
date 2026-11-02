"""One request budget for production protocols and their isolated evaluators."""

PROTOCOL_RESERVE = 150


def reply_request_params(params, *, native=False, core_enabled=False):
    """Expand the base budget once, preserving the native protocol's existing reserve.

    The allowance is shared by generated fields, not a hard Response reservation.
    This function neither clips text nor triggers a retry.
    """
    request = dict(params)
    if native or core_enabled:
        request["max_tokens"] = int(params.get("max_tokens", 200)) + PROTOCOL_RESERVE
    return request
