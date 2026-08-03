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

"""Additional tasks for recipes app."""

import dataclasses
import random
from typing import Any

from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import recipe
from android_world.task_evals.single.recipe import (
    _APP_NAME,
    _RecipeAddMultipleRecipes,
    _RecipeDeleteMultipleRecipes,
    _generate_random_recipe,
    _get_rows_as_text,
    _TEXT_REPRESENTATION_TYPE,
    _PREP_TIME_OPTIONS,
    _SERVINGS_OPTIONS,
)
from android_world.task_evals.utils import sqlite_schema_utils


class RecipeAddFavoriteRecipe(_RecipeAddMultipleRecipes):
  """Task to add a recipe and explicitly mark it as a favorite."""

  complexity = 3.2
  n_rows = 1
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    text_repr = _get_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return (
        f'Add the following recipe to {_APP_NAME} and make sure to mark it as a'
        f' favorite:\n{text_repr}'
    )

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    row = _generate_random_recipe()
    return dataclasses.replace(row, favorite=True)


class RecipeDeleteFavoriteRecipes(_RecipeDeleteMultipleRecipes):
  """Task to delete all recipes that are marked as favorites."""

  complexity = 2.5
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    return f'Delete all recipes marked as "Favorites" from the {_APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Generate target favorites
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_random_recipe()
        targets.append(dataclasses.replace(row, favorite=True))
    
    # Generate noise (non-favorites)
    noise = []
    for _ in range(cls.n_rows_noise):
        row = _generate_random_recipe()
        noise.append(dataclasses.replace(row, favorite=False))
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeAddQuickMeal(_RecipeAddMultipleRecipes):
  """Task to add a recipe with a short preparation time (10 mins)."""

  complexity = 3
  n_rows = 1
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    row = _generate_random_recipe()
    return dataclasses.replace(row, preparationTime='10 mins')

  @property
  def goal(self) -> str:
    text_repr = _get_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return (
        f'Add this quick meal (10 mins prep) to {_APP_NAME}:\n{text_repr}'
    )


class RecipeDeleteLongPrepRecipes(_RecipeDeleteMultipleRecipes):
  """Task to delete recipes that take a long time to prepare (>= 3 hrs)."""

  complexity = 3.5
  n_rows = 2
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return (
        f'In {_APP_NAME}, delete all recipes that take 3 hours or longer to'
        ' prepare.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    long_times = ['3 hrs', '4 hrs']
    short_times = [t for t in _PREP_TIME_OPTIONS if t not in long_times]
    
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_random_recipe()
        targets.append(dataclasses.replace(row, preparationTime=random.choice(long_times)))
        
    noise = []
    for _ in range(cls.n_rows_noise):
        row = _generate_random_recipe()
        noise.append(dataclasses.replace(row, preparationTime=random.choice(short_times)))
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeAddPartyRecipe(_RecipeAddMultipleRecipes):
  """Task to add a recipe designed for a large group (8 servings)."""

  complexity = 3
  n_rows = 1
  n_rows_noise = 5

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    row = _generate_random_recipe()
    return dataclasses.replace(row, servings='8 servings')


class RecipeDeleteSoloMeals(_RecipeDeleteMultipleRecipes):
  """Task to delete all recipes serving only 1 person."""

  complexity = 2.5
  n_rows = 2
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Remove all recipes for "1 serving" from {_APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_random_recipe()
        targets.append(dataclasses.replace(row, servings='1 serving'))
        
    noise = []
    # Ensure noise is not 1 serving
    other_servings = [s for s in _SERVINGS_OPTIONS if s != '1 serving']
    for _ in range(cls.n_rows_noise):
        row = _generate_random_recipe()
        noise.append(dataclasses.replace(row, servings=random.choice(other_servings)))
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeAddSourceRecipe(_RecipeAddMultipleRecipes):
  """Task to add a recipe with a specific source citation."""

  complexity = 3.5
  n_rows = 1
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    row = _generate_random_recipe()
    return dataclasses.replace(row, source='Grandma\'s Secret Cookbook')

  @property
  def goal(self) -> str:
    text_repr = _get_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return (
        f'Add the following recipe to {_APP_NAME}, ensuring the Source field is'
        f' set to "Grandma\'s Secret Cookbook":\n{text_repr}'
    )


class RecipeDeleteRecipesFromSource(_RecipeDeleteMultipleRecipes):
  """Task to delete all recipes from a specific source."""

  complexity = 3.0
  n_rows = 2
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    return (
        f'Delete all recipes sourced from "{self.params["source"]}" in {_APP_NAME}.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    target_source = "Internet Archive"
    noise_source = "My Own Creations"
    
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_random_recipe()
        targets.append(dataclasses.replace(row, source=target_source))
        
    noise = []
    for _ in range(cls.n_rows_noise):
        row = _generate_random_recipe()
        noise.append(dataclasses.replace(row, source=noise_source))
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        "source": target_source
    }


class RecipeAddHighProteinRecipe(_RecipeAddMultipleRecipes):
  """Task to add a recipe that explicitly mentions 'High Protein' in the description."""

  complexity = 3
  n_rows = 1
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    row = _generate_random_recipe()
    new_desc = f"High Protein. {row.description}"
    return dataclasses.replace(row, description=new_desc)


class RecipeDeleteRecipesWithKeyword(_RecipeDeleteMultipleRecipes):
  """Task to delete recipes containing a specific keyword in the title."""

  complexity = 3.0
  n_rows = 2
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return (
        f'Delete all recipes with the word "{self.params["keyword"]}" in the'
        f' title from {_APP_NAME}.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    keyword = "Spicy"
    
    # Generate targets with keyword
    targets = []
    while len(targets) < cls.n_rows:
        row = _generate_random_recipe()
        if keyword not in row.title:
            new_title = f"{keyword} {row.title}"
            row = dataclasses.replace(row, title=new_title)
        targets.append(row)

    # Generate noise without keyword
    noise = []
    while len(noise) < cls.n_rows_noise:
        row = _generate_random_recipe()
        if keyword not in row.title:
            noise.append(row)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        "keyword": keyword
    }