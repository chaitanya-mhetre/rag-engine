import json
import logging

from ragengine.observability import JsonFormatter, request_id_var


def test_json_formatter_includes_request_id_and_extras() -> None:
    token = request_id_var.set("abc")
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("w",), None)
        record.status = 200
        data = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert data["msg"] == "hello w" and data["request_id"] == "abc" and data["status"] == 200
