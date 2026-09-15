"""服务端执行、绑定清单的评测 Fixture 校验（自单文件拆分为包）。

校验证据刻意只保留持久标识符、原因码与时间戳，绝不存储测试身份配置、
文档正文、检索上下文、凭据、租户覆盖或不受限的问答结果。
"""

from evaluation.fixture_validation.assertions import (
    FixtureAssertionRuntime,
    build_catalog_fixture_corpus_probe,
)
from evaluation.fixture_validation.store import (
    FIXTURE_VALIDATION_SCHEMA,
    FIXTURE_VALIDATION_STATUSES,
    REQUIRED_FIXTURES,
    FinanceFixtureIdentityResolver,
    FixtureExecutionIdentity,
    FixtureValidationConflict,
    FixtureValidationError,
    FixtureValidationStore,
)
from evaluation.fixture_validation.service import FixtureValidationService

__all__ = [
    "FIXTURE_VALIDATION_SCHEMA",
    "FIXTURE_VALIDATION_STATUSES",
    "REQUIRED_FIXTURES",
    "FinanceFixtureIdentityResolver",
    "FixtureAssertionRuntime",
    "FixtureExecutionIdentity",
    "FixtureValidationConflict",
    "FixtureValidationError",
    "FixtureValidationService",
    "FixtureValidationStore",
    "build_catalog_fixture_corpus_probe",
]
