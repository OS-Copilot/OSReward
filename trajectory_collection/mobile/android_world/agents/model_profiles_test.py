# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for model profiles.

The load-bearing test here is `test_extract_history_matches_own_system_prompt`.
A `ModelProfile` pairs a system prompt with a history extractor, but the two are
independent fields with no structural link, so they can silently drift apart --
which is exactly what happened to the `qwen3vl` profile: its prompt asked for
`<conclusion>` while its extractor scanned for an `Action:` line, so every step
summary came back empty and the agent ran with no memory of its own actions.
"""

from absl.testing import absltest
from absl.testing import parameterized

from android_world.agents import model_profiles


def _example_response(system_prompt: str) -> str:
  """Returns the literal example response embedded in a system prompt.

  Each system prompt ends with an `Example:` section demonstrating the exact
  output format it asks the model to produce. That block is the best available
  stand-in for a real response, and -- unlike a hand-written fixture -- it
  tracks the prompt automatically when the prompt is edited.
  """
  head, sep, example = system_prompt.partition("Example:")
  del head
  if not sep:
    raise AssertionError(
        "System prompt has no 'Example:' section, so its output format cannot"
        " be round-tripped through its extractor. Add an example to the prompt,"
        " or update this test."
    )
  return example.strip()


class ModelProfilesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (profile_id, profile_id) for profile_id in model_profiles.PROFILES
  )
  def test_extract_history_matches_own_system_prompt(self, profile_id: str):
    """Each profile's extractor must handle the format its own prompt asks for.

    This is the regression guard for the `qwen3vl` mispairing. Feeding the
    prompt's own example through the profile's extractor must yield a non-empty
    summary; an empty result means the agent would accumulate a step history of
    "Step 1: ; Step 2: ; ..." and lose all task progress context.
    """
    profile = model_profiles.PROFILES[profile_id]
    example = _example_response(profile.system_prompt)

    summary = profile.extract_history(example)

    self.assertNotEmpty(
        summary,
        f"Profile {profile_id!r} extracted an empty step summary from the"
        " example in its own system prompt. Its `extract_history` does not"
        " match the format the prompt requests, so the agent will run with no"
        " memory of its previous actions.",
    )

  @parameterized.named_parameters(
      (profile_id, profile_id) for profile_id in model_profiles.PROFILES
  )
  def test_extract_history_is_single_line(self, profile_id: str):
    """Summaries are concatenated into a one-line history string."""
    profile = model_profiles.PROFILES[profile_id]
    summary = profile.extract_history(_example_response(profile.system_prompt))
    self.assertNotIn("\n", summary)

  @parameterized.named_parameters(
      (profile_id, profile_id) for profile_id in model_profiles.PROFILES
  )
  def test_extract_history_tolerates_empty_response(self, profile_id: str):
    """A filtered or empty response must not raise, just yield nothing."""
    profile = model_profiles.PROFILES[profile_id]
    self.assertEqual(profile.extract_history(""), "")

  def test_every_mapped_model_resolves(self):
    for model_name in model_profiles.MODEL_TO_PROFILE:
      self.assertIsInstance(
          model_profiles.resolve(model_name), model_profiles.ModelProfile
      )

  def test_profile_ids_match_their_keys(self):
    for key, profile in model_profiles.PROFILES.items():
      self.assertEqual(key, profile.profile_id)

  def test_unknown_model_raises(self):
    with self.assertRaises(ValueError):
      model_profiles.resolve("not-a-real-model")

  def test_empty_model_raises(self):
    with self.assertRaises(ValueError):
      model_profiles.resolve("")

  def test_unknown_profile_override_raises(self):
    with self.assertRaises(ValueError):
      model_profiles.resolve("gemini-3-flash", profile_override="nope")

  def test_profile_override_wins_over_model_name(self):
    profile = model_profiles.resolve(
        "gemini-3-flash", profile_override="qwen3vl"
    )
    self.assertEqual(profile.profile_id, "qwen3vl")


class ExtractorsTest(absltest.TestCase):

  def test_conclusion_preferred_over_thinking(self):
    block = (
        "<thinking>reasoning here</thinking>"
        "<tool_call>{}</tool_call>"
        "<conclusion>the summary</conclusion>"
    )
    self.assertEqual(
        model_profiles._conclusion_then_thinking(block), "the summary"
    )

  def test_falls_back_to_thinking_when_conclusion_absent(self):
    block = "<thinking>reasoning here</thinking><tool_call>{}</tool_call>"
    self.assertEqual(
        model_profiles._conclusion_then_thinking(block), "reasoning here"
    )

  def test_returns_empty_when_neither_tag_present(self):
    self.assertEqual(
        model_profiles._conclusion_then_thinking("<tool_call>{}</tool_call>"), ""
    )

  def test_action_text_does_not_match_conclusion_format(self):
    """Documents the bug this module previously shipped.

    `extract_action_text` returns nothing for a `<conclusion>`-style response.
    Pairing it with a prompt that asks for `<conclusion>` is what emptied the
    Qwen3-VL step history.
    """
    block = (
        "<thinking>reasoning</thinking>"
        "<tool_call>{}</tool_call>"
        "<conclusion>the summary</conclusion>"
    )
    self.assertEqual(model_profiles.extract_action_text(block), "")


if __name__ == "__main__":
  absltest.main()
