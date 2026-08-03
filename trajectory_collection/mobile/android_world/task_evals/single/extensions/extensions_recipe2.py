"""Additional complex tasks for Broccoli Recipe App."""

import dataclasses
import random
from typing import Any
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import recipe
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils

# Extended options for generation
_FAVORITE_OPTIONS = [True, False]


class RecipeDeleteByPrepTime(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes that take a specific amount of time to prepare."""

  complexity = 3.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    target_time = self.params['target_time']
    return (
        f'In the {recipe._APP_NAME}, delete all recipes that take exactly'
        f' "{target_time}" to prepare.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    target_time = random.choice(recipe._PREP_TIME_OPTIONS)
    
    # Generate targets with the specific time
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        recipe._generate_random_recipe,
        replacement=False,
    )
    targets = [dataclasses.replace(r, preparationTime=target_time) for r in targets]

    # Generate noise with different times
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.preparationTime != target_time
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        'target_time': target_time,
    }


class RecipeDeleteByServings(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes designed for a specific number of servings."""

  complexity = 3.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    servings = self.params['servings']
    return f'Delete all recipes serving "{servings}" from the {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    servings = random.choice(recipe._SERVINGS_OPTIONS)
    
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, servings=servings) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.servings != servings
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        'servings': servings,
    }


class RecipeDeleteFavorites(recipe._RecipeDeleteMultipleRecipes):
  """Delete all recipes marked as favorites."""

  complexity = 2.5
  max_steps = 20
  n_rows = 4
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Remove all favorite recipes from the {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, favorite=True) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
    )
    noise = [dataclasses.replace(r, favorite=False) for r in noise]

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteNonFavorites(recipe._RecipeDeleteMultipleRecipes):
  """Delete all recipes that are NOT favorites."""

  complexity = 3.0
  max_steps = 25
  n_rows = 4
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Clean up the {recipe._APP_NAME} by deleting all non-favorite recipes.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, favorite=False) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
    )
    noise = [dataclasses.replace(r, favorite=True) for r in noise]

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteByDescriptionKeyword(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes containing a specific keyword in the description."""

  complexity = 3.5
  max_steps = 30
  n_rows = 3
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return f'Delete any recipes describing themselves as "healthy" from {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    keyword = "healthy"
    
    # Generate targets containing keyword
    targets = []
    while len(targets) < cls.n_rows:
      cand = recipe._generate_random_recipe()
      if keyword.lower() in cand.description.lower():
         targets.append(cand)
      else:
         # Force keyword injection
         cand = dataclasses.replace(cand, description=f"{cand.description} It is very {keyword}.")
         targets.append(cand)

    # Generate noise NOT containing keyword
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: keyword.lower() not in r.description.lower()
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteByTitleSubstring(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes with a specific substring in the title."""

  complexity = 2.5
  max_steps = 20
  n_rows = 3
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return f'Delete all "Chicken" recipes from the {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    keyword = "Chicken"
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: keyword in r.title
    )
    # Ensure we have enough targets, force if necessary (though Chicken is common in the list)
    if len(targets) < cls.n_rows:
        extra_needed = cls.n_rows - len(targets)
        extras = sqlite_schema_utils.get_random_items(extra_needed, recipe._generate_random_recipe, replacement=True)
        extras = [dataclasses.replace(r, title=f"{keyword} {r.title}") for r in extras]
        targets.extend(extras)

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: keyword not in r.title
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteEmptyIngredients(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes where the ingredients list is unhelpful/empty."""

  complexity = 3.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    return f'Delete recipes where the ingredient list is just "n/a" in {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, ingredients="n/a") for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.ingredients != "n/a"
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteQuickRecipes(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes that take less than 20 minutes."""

  complexity = 3.5
  max_steps = 25
  n_rows = 4
  n_rows_noise = 12

  @property
  def goal(self) -> str:
    return f'Delete all quick recipes (10 or 20 mins) from {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    quick_times = ['10 mins', '20 mins']
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, preparationTime=random.choice(quick_times)) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.preparationTime not in quick_times
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeDeleteFeastRecipes(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes for large groups (6 or 8 servings)."""

  complexity = 3.5
  max_steps = 25
  n_rows = 3
  n_rows_noise = 12

  @property
  def goal(self) -> str:
    return f'Delete all recipes serving 6 or 8 people from {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    large_servings = ['6 servings', '8 servings']
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, servings=random.choice(large_servings)) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.servings not in large_servings
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeAddMarkorHealthy(recipe.RecipeAddMultipleRecipesFromMarkor):
  """Add only recipes containing 'Healthy' in the description from text file."""

  complexity = 5.0
  max_steps = 40
  n_rows = 3
  n_rows_noise = 30

  @property
  def goal(self) -> str:
    return (
        f'Parse recipes.txt in Markor and add only the recipes described as "Healthy" '
        f'to the {recipe._APP_NAME}.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets must have "Healthy"
    targets = sqlite_schema_utils.get_random_items(cls.n_rows, recipe._generate_random_recipe)
    targets = [dataclasses.replace(r, description=f"Healthy choice. {r.description}") for r in targets]

    # Noise must NOT have "Healthy"
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise, 
        recipe._generate_random_recipe, 
        filter_fn=lambda r: "Healthy" not in r.description
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        recipe._TEXT_REPRESENTATION_TYPE: 'text_block',
    }


class RecipeAddMarkorByIngredient(recipe.RecipeAddMultipleRecipesFromMarkor):
  """Add recipes containing a specific ingredient from text file."""

  complexity = 5.0
  max_steps = 40
  n_rows = 3
  n_rows_noise = 30

  @property
  def goal(self) -> str:
    ingredient = self.params['ingredient']
    return (
        f'From the recipes.txt file in Markor, add only the recipes that contain '
        f'{ingredient} to the {recipe._APP_NAME}.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    ingredient = random.choice(recipe._COMMON_INGREDIENTS)
    
    # Targets: Contain ingredient
    targets = []
    while len(targets) < cls.n_rows:
        r = recipe._generate_random_recipe()
        if ingredient.lower() in r.ingredients.lower() or ingredient.lower() in r.directions.lower():
            targets.append(r)
        else:
            targets.append(dataclasses.replace(r, ingredients=f"{r.ingredients}, {ingredient}"))
            
    # Noise: Do not contain ingredient
    noise = []
    while len(noise) < cls.n_rows_noise:
        r = recipe._generate_random_recipe()
        if ingredient.lower() not in r.ingredients.lower() and ingredient.lower() not in r.directions.lower():
            noise.append(r)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        recipe._TEXT_REPRESENTATION_TYPE: 'text_block',
        'ingredient': ingredient,
    }


class RecipeAddMarkorFavorites(recipe.RecipeAddMultipleRecipesFromMarkor):
  """Add recipes from file and mark them as Favorite during addition."""

  complexity = 5.5
  max_steps = 40
  n_rows = 3
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return (
        f'Import all recipes from recipes.txt in Markor to {recipe._APP_NAME} '
        'and ensure they are marked as Favorites.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    # The file contains the rows, but the validator expects them to be favorite=True in DB.
    # We must ensure the file content matches the "before-favorite" state if needed,
    # but simplest is just writing the full objects to text.
    # However, the task implies the USER marks them as favorite, not that the file says "Favorite: Yes".
    # So we'll write them to file without explicit favorite text if possible, or just standard.
    # Base class implementation writes whatever is in ROW_OBJECTS.
    # We need to hack the initialization to write non-favorite versions to file,
    # but expect favorite versions in DB.
    
    # 1. Save original expected targets (which have favorite=True)
    expected_targets = self.params[sqlite_validators.ROW_OBJECTS]
    
    # 2. Create file versions (favorite=False or unspecified)
    file_targets = [dataclasses.replace(r, favorite=False) for r in expected_targets]
    
    # 3. Use base class init but swap params temporarily
    noise = self.params[sqlite_validators.NOISE_ROW_OBJECTS]
    rows = file_targets + noise
    random.shuffle(rows)
    
    file_utils.clear_directory(device_constants.MARKOR_DATA, env.controller)
    user_data_generation.write_to_markor(
        recipe._get_rows_as_text(rows, self.params[recipe._TEXT_REPRESENTATION_TYPE]),
        'recipes.txt',
        env,
    )
    # No need to call super().initialize_task as we did the work.

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = sqlite_schema_utils.get_random_items(cls.n_rows, recipe._generate_random_recipe)
    targets = [dataclasses.replace(r, favorite=True) for r in targets]
    
    # Noise for the file (not added to DB)
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, recipe._generate_random_recipe)
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        recipe._TEXT_REPRESENTATION_TYPE: 'text_block',
    }


class RecipeAddLargeBatch(recipe._RecipeAddMultipleRecipes):
  """Add a larger batch of recipes manually."""

  complexity = 5.0
  max_steps = 50
  n_rows = 5
  n_rows_noise = 10


class RecipeDeduplicateByTitle(recipe.RecipeDeleteDuplicateRecipes):
  """Deduplicate based strictly on title, ignoring description differences."""

  complexity = 3.0
  max_steps = 25
  n_rows = 1
  n_rows_noise = 20

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    target_base = recipe._generate_random_recipe()
    
    # Target 1: Original
    t1 = target_base
    # Target 2: Same title, different description
    t2 = dataclasses.replace(target_base, description="A different description for the same dish.")
    
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.title != target_base.title
    )
    
    return {
        sqlite_validators.ROW_OBJECTS: [t1, t2],
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class RecipeAddDetailed(recipe._RecipeAddMultipleRecipes):
  """Add a recipe with maximum detail/length in fields."""

  complexity = 4.0
  max_steps = 25
  n_rows = 1
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Recipe:
    r = recipe._generate_random_recipe()
    long_desc = "This is a very detailed description that goes on for quite a while to test the input capabilities. " * 3
    long_dir = "Step 1: Do this. Step 2: Do that. " * 10
    return dataclasses.replace(r, description=long_desc, directions=long_dir)


class RecipeDeleteBySource(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes from a specific source."""

  complexity = 3.0
  max_steps = 20
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    source = self.params['source']
    return f'Delete all recipes sourced from "{source}" in {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    source = "Family Cookbook"
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, replacement=False
    )
    targets = [dataclasses.replace(r, source=source) for r in targets]

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        recipe._generate_random_recipe,
        replacement=False,
        filter_fn=lambda r: r.source != source
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        'source': source,
    }


class RecipeAddMultipleRecipesFromMarkorQuick(recipe.RecipeAddMultipleRecipesFromMarkor):
  """Add only quick recipes from text file."""

  complexity = 5.0
  max_steps = 40
  n_rows = 3
  n_rows_noise = 30

  @property
  def goal(self) -> str:
    return (
        f'Import recipes from recipes.txt in Markor that take 20 mins or less '
        f'to {recipe._APP_NAME}.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    quick_times = ['10 mins', '20 mins']
    
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, filter_fn=lambda r: r.preparationTime in quick_times
    )
    
    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise, recipe._generate_random_recipe, filter_fn=lambda r: r.preparationTime not in quick_times
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        recipe._TEXT_REPRESENTATION_TYPE: 'text_block',
    }


class RecipeDeleteTitleStartsWith(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes starting with a specific letter."""

  complexity = 3.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    letter = self.params['letter']
    return f'Delete all recipes starting with the letter "{letter}" in {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    letter = 'S' # 'S' is common enough (Salad, Spicy, etc.)
    
    targets = sqlite_schema_utils.get_random_items(
        cls.n_rows, recipe._generate_random_recipe, filter_fn=lambda r: r.title.startswith(letter)
    )
    
    # If not enough natural ones, force rename
    while len(targets) < cls.n_rows:
        r = recipe._generate_random_recipe()
        r = dataclasses.replace(r, title=f"Super {r.title}")
        targets.append(r)

    noise = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise, recipe._generate_random_recipe, filter_fn=lambda r: not r.title.startswith(letter)
    )

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        'letter': letter,
    }


class RecipeDeleteMultipleRecipesComplexLogic(recipe._RecipeDeleteMultipleRecipes):
  """Delete recipes that match complex logic: Healthy AND Quick."""

  complexity = 4.5
  max_steps = 35
  n_rows = 2
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return f'Delete recipes that are both "Healthy" (in description) AND take "10 mins" in {recipe._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets: Healthy + 10 mins
    targets = []
    while len(targets) < cls.n_rows:
        r = recipe._generate_random_recipe()
        r = dataclasses.replace(r, description=f"Healthy. {r.description}", preparationTime="10 mins")
        targets.append(r)
        
    # Noise 1: Healthy but slow
    noise1 = [dataclasses.replace(recipe._generate_random_recipe(), description="Healthy option.", preparationTime="1 hrs") for _ in range(5)]
    # Noise 2: Quick but not healthy
    noise2 = [dataclasses.replace(recipe._generate_random_recipe(), description="Indulgent.", preparationTime="10 mins") for _ in range(5)]
    # Noise 3: Neither
    noise3 = sqlite_schema_utils.get_random_items(10, recipe._generate_random_recipe)
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise1 + noise2 + noise3,
    }