import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from library.bhsa import (
    DEFAULT_BHSA_CLONE,
    bhsa_clone_location,
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
    load_bhsa_api,
    node_to_psalm_map,
    psalms_book_node,
)


def _stub_api() -> SimpleNamespace:
    """An object shaped enough like a Text-Fabric api to pass the loader's api check."""
    return SimpleNamespace(F=object())


def _fake_fabric_class(api: object):
    """A fabric_class stand-in whose loadAll() always returns the given fake api."""

    def fake_fabric_class(*, locations: list[str], silent: str) -> object:
        class _FakeTF:
            def loadAll(self, silent: str) -> object:  # noqa: N802
                return api

        return _FakeTF()

    return fake_fabric_class


def _load_expecting_clone_failure(**kwargs: object) -> object:
    """Drives the local-clone failure path, asserting the diagnostic that path must emit."""
    with pytest.warns(RuntimeWarning, match="falling back to use"):
        return load_bhsa_api(**kwargs)  # type: ignore[arg-type]


def _raising_fabric_class(exc: Exception):
    """A fabric_class stand-in whose construction raises, simulating a missing local clone."""

    def fake_fabric_class(*, locations: list[str], silent: str) -> object:
        raise exc

    return fake_fabric_class


class _FakeOtype:
    def __init__(self, book_types: dict[int, str]) -> None:
        self._book_types = book_types

    def s(self, otype: str) -> list[int]:
        assert otype == "book"
        return list(self._book_types)


class _FakeFeature:
    def __init__(self, values: dict[int, str]) -> None:
        self._values = values

    def v(self, node: int) -> str | None:
        return self._values.get(node)


class _FakeF:
    def __init__(self, book_names: dict[int, str]) -> None:
        self.otype = _FakeOtype(book_names)
        self.book = _FakeFeature(book_names)


class _FakeL:
    def __init__(self, children: dict[tuple[int, str], list[int]]) -> None:
        self._children = children

    def d(self, node: int, otype: str) -> list[int]:
        return self._children.get((node, otype), [])


class _FakeT:
    def __init__(self, chapter_to_psalm: dict[int, int]) -> None:
        self._chapter_to_psalm = chapter_to_psalm

    def sectionFromNode(self, node: int) -> tuple[str, int]:  # noqa: N802
        return ("Psalmi", self._chapter_to_psalm[node])


class _FakeApi:
    def __init__(self, F: _FakeF, L: _FakeL, T: _FakeT) -> None:  # noqa: N803
        self.F = F
        self.L = L
        self.T = T


def _api_with_two_psalms() -> _FakeApi:
    book_names = {1: "Psalmi"}
    children = {
        (1, "chapter"): [10, 20],
        (10, "half_verse"): [100, 101],
        (20, "half_verse"): [200],
    }
    chapter_to_psalm = {10: 1, 20: 2}
    return _FakeApi(_FakeF(book_names), _FakeL(children), _FakeT(chapter_to_psalm))


class TestLoadBhsaApi:
    def test_returns_the_api_from_a_successful_local_clone_load(self) -> None:
        local_api = _stub_api()

        result = load_bhsa_api(fabric_class=_fake_fabric_class(local_api))

        assert result is local_api

    def test_does_not_call_use_when_the_local_clone_succeeds(self) -> None:
        local_api = _stub_api()
        use_calls = []

        def tracking_use(*a: object, **k: object) -> object:
            use_calls.append((a, k))
            raise AssertionError("use() should not be called when the local clone succeeds")

        load_bhsa_api(use_fn=tracking_use, fabric_class=_fake_fabric_class(local_api))

        assert use_calls == []

    def test_falls_back_to_use_when_the_local_clone_raises(self) -> None:
        fallback_api = _stub_api()

        class _FakeApp:
            api = fallback_api

        with pytest.warns(RuntimeWarning, match="falling back to use"):
            result = load_bhsa_api(
                use_fn=lambda *a, **k: _FakeApp(),
                fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
                timeout_seconds=1.0,
            )

        assert result is fallback_api

    def test_falls_back_to_use_when_the_local_loadall_returns_none(self) -> None:
        fallback_api = _stub_api()

        class _FakeApp:
            api = fallback_api

        result = load_bhsa_api(
            use_fn=lambda *a, **k: _FakeApp(),
            fabric_class=_fake_fabric_class(None),
            timeout_seconds=1.0,
        )

        assert result is fallback_api

    def test_passes_checkout_and_mod_through_to_use_during_fallback(self) -> None:
        calls = []

        def fake_use(name: str, *, checkout: str, mod: str | None, silent: str) -> object:
            calls.append((name, checkout, mod))

            class _FakeApp:
                api = _stub_api()

            return _FakeApp()

        _load_expecting_clone_failure(
            checkout="v1.0",
            mod="org/repo/tf:v1.0",
            use_fn=fake_use,
            fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
        )

        assert calls == [("etcbc/bhsa", "v1.0", "org/repo/tf:v1.0")]

    def test_falls_back_to_use_and_waits_up_to_the_timeout_when_the_local_clone_fails(
        self,
    ) -> None:
        fallback_api = _stub_api()

        def slow_but_in_time_use(*a: object, **k: object) -> object:
            time.sleep(0.05)

            class _FakeApp:
                api = fallback_api

            return _FakeApp()

        result = _load_expecting_clone_failure(
            use_fn=slow_but_in_time_use,
            fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
            timeout_seconds=1.0,
        )

        assert result is fallback_api

    def test_does_not_wait_for_a_slow_use_call_past_the_timeout(self) -> None:
        started = time.monotonic()

        def slow_use(*a: object, **k: object) -> object:
            time.sleep(0.3)
            return None

        with (
            pytest.warns(RuntimeWarning, match="falling back to use"),
            pytest.raises(RuntimeError, match="Text-Fabric failed"),
        ):
            load_bhsa_api(
                use_fn=slow_use,
                fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
                timeout_seconds=0.05,
            )

        assert time.monotonic() - started < 0.2

    def test_passes_the_mod_org_and_repo_to_the_local_clone_locations(self) -> None:
        captured_locations = []
        captured_mods = []

        def fake_fabric_class(*, locations: list[str], silent: str) -> object:
            captured_locations.append(locations)

            class _FakeTF:
                def loadAll(self, silent: str) -> object:  # noqa: N802
                    return SimpleNamespace(F=object())

            return _FakeTF()

        def fake_mod_cache_location_fn(mod: str) -> Path:
            captured_mods.append(mod)
            return Path("/fake/mod/cache/location")

        load_bhsa_api(
            mod="rdtaylorjr/tehillim-logos/tf:v1.0",
            fabric_class=fake_fabric_class,
            mod_cache_location_fn=fake_mod_cache_location_fn,
        )

        assert captured_mods == ["rdtaylorjr/tehillim-logos/tf:v1.0"]
        assert len(captured_locations) == 1
        assert len(captured_locations[0]) == 2

    def test_raises_when_both_the_local_clone_and_use_fail(self) -> None:
        with (
            pytest.warns(RuntimeWarning, match="falling back to use"),
            pytest.raises(RuntimeError, match="Text-Fabric failed"),
        ):
            load_bhsa_api(
                use_fn=lambda *a, **k: None,
                fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
                timeout_seconds=1.0,
            )


class TestPsalmsBookNode:
    def test_finds_the_psalms_book_node(self) -> None:
        api = _FakeApi(_FakeF({5: "Psalmi", 6: "Genesis"}), _FakeL({}), _FakeT({}))

        assert psalms_book_node(api) == 5

    def test_raises_when_no_psalms_book_is_present(self) -> None:
        api = _FakeApi(_FakeF({6: "Genesis"}), _FakeL({}), _FakeT({}))

        with pytest.raises(RuntimeError, match="Psalmi"):
            psalms_book_node(api)


class TestListPsalmsHalfVerseNodes:
    def test_lists_every_half_verse_node_across_all_chapters(self) -> None:
        api = _api_with_two_psalms()

        assert list_psalms_half_verse_nodes(api) == [100, 101, 200]


class TestListPsalmsHalfVersesByPsalm:
    def test_groups_half_verse_nodes_by_psalm_number(self) -> None:
        api = _api_with_two_psalms()

        assert list_psalms_half_verses_by_psalm(api) == {1: [100, 101], 2: [200]}


class TestNodeToPsalmMap:
    def test_inverts_a_psalm_to_nodes_mapping(self) -> None:
        result = node_to_psalm_map({1: [100, 101], 2: [200]})

        assert result == {100: 1, 101: 1, 200: 2}


class TestLocalCloneDiagnostics:
    def test_warns_with_the_reason_when_the_local_clone_fails(self) -> None:
        """A swallowed failure looked identical to a cold cache, hiding real loader bugs."""

        class _FakeApp:
            api = _stub_api()

        with pytest.warns(RuntimeWarning, match="no local clone"):
            load_bhsa_api(
                use_fn=lambda *a, **k: _FakeApp(),
                fabric_class=_raising_fabric_class(RuntimeError("no local clone")),
                timeout_seconds=1.0,
            )

    def test_does_not_warn_when_the_local_clone_succeeds(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_bhsa_api(fabric_class=_fake_fabric_class(_stub_api()))

    def test_reads_the_clone_from_the_injected_location(self, tmp_path: Path) -> None:
        """The clone path is machine-specific, so it is a parameter, not a baked-in constant."""
        seen: list[list[str]] = []

        def recording_fabric_class(*, locations: list[str], silent: str) -> object:
            seen.append(locations)
            return _fake_fabric_class(_stub_api())(locations=locations, silent=silent)

        load_bhsa_api(fabric_class=recording_fabric_class, clone_location=tmp_path)

        assert seen == [[str(tmp_path)]]


class TestBhsaCloneLocation:
    def test_prefers_the_environment_variable_when_it_is_set(self) -> None:
        location = bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": "/custom/bhsa/tf/2021"})

        assert location == Path("/custom/bhsa/tf/2021")

    def test_falls_back_to_the_conventional_clone_path_when_unset(self) -> None:
        assert bhsa_clone_location(env={}) == DEFAULT_BHSA_CLONE

    def test_treats_an_empty_variable_as_unset(self) -> None:
        assert bhsa_clone_location(env={"TEHILLIM_BHSA_PATH": ""}) == DEFAULT_BHSA_CLONE

    def test_load_bhsa_api_reads_the_environment_when_no_location_is_passed(self, tmp_path) -> None:
        seen: list[list[str]] = []

        class RecordingFabric:
            def __init__(self, locations, silent):
                seen.append(locations)

            def loadAll(self, silent):  # noqa: N802 -- Text-Fabric's own method name
                return SimpleNamespace(F=object())

        load_bhsa_api(fabric_class=RecordingFabric, env={"TEHILLIM_BHSA_PATH": str(tmp_path)})

        assert seen == [[str(tmp_path)]]


class TestNonApiLocalResults:
    def test_falls_back_when_the_local_clone_returns_false(self) -> None:
        """A failed load is reported as False, which is not None and must not be returned."""

        class FalseFabric:
            def __init__(self, locations, silent):
                pass

            def loadAll(self, silent):  # noqa: N802 -- Text-Fabric's own method name
                return False

        sentinel = SimpleNamespace(F=object())

        api = load_bhsa_api(
            fabric_class=FalseFabric,
            use_fn=lambda *a, **k: SimpleNamespace(api=sentinel),
        )

        assert api is sentinel

    def test_raises_when_both_sides_return_a_non_api(self) -> None:
        class FalseFabric:
            def __init__(self, locations, silent):
                pass

            def loadAll(self, silent):  # noqa: N802 -- Text-Fabric's own method name
                return False

        with pytest.raises(RuntimeError, match="failed to load BHSA"):
            load_bhsa_api(
                fabric_class=FalseFabric,
                use_fn=lambda *a, **k: SimpleNamespace(api=False),
            )
