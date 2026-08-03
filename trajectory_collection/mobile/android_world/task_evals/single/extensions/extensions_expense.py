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

"""Additional tasks for managing expenses in an expense app."""

import dataclasses
import random
from typing import Any

from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import expense
from android_world.task_evals.single.expense import (
    _APP_NAME,
    _TEXT_REPRESENTATION_TYPE,
    _ExpenseAddMultiple,
    _ExpenseDeleteMultiple,
    _generate_expense,
    _get_expense_rows_as_text,
)
from android_world.task_evals.utils import sqlite_schema_utils


class ExpenseAddHighValueSingle(_ExpenseAddMultiple):
  """Task to add a single high-value expense (over $500)."""

  complexity = 1.4
  n_rows = 1
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    row = _generate_expense()
    # Set amount between $500 and $2000
    amount = random.randint(50000, 200000)
    return dataclasses.replace(row, amount=amount)


class ExpenseAddMultipleFoodCategory(_ExpenseAddMultiple):
  """Task to add multiple expenses specifically for the Food category."""

  complexity = 3
  n_rows = 3
  n_rows_noise = 10

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    # Find the ID for 'Food'
    food_id = next(
        k
        for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
        if v == 'Food'
    )
    return _generate_expense(category_id=food_id)


class ExpenseAddVacationExpenses(_ExpenseAddMultiple):
  """Task to add a group of expenses simulating a vacation (Transportation, Hotel, Food)."""

  complexity = 3.2
  n_rows = 3
  n_rows_noise = 10

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Manually construct specific categories for a vacation
    categories = ['Transportation', 'Housing', 'Food']
    target_rows = []
    for cat_name in categories:
        cat_id = next(
            k
            for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
            if v == cat_name
        )
        target_rows.append(_generate_expense(category_id=cat_id))
    
    # Generate noise
    noise_rows = sqlite_schema_utils.get_random_items(
        cls.n_rows_noise,
        _generate_expense,
        replacement=False,
        filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    )
    
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        _TEXT_REPRESENTATION_TYPE: 'text_block',
    }


class ExpenseAddWithSpecificNote(_ExpenseAddMultiple):
  """Task to add a single expense with a very specific, custom note."""

  complexity = 1.5
  n_rows = 1
  n_rows_noise = 5

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    row = _generate_expense()
    custom_notes = [
        "Reimbursement pending from office",
        "Split bill with John and Sarah",
        "Warranty expires in 2 years",
        "Gift for mom's 60th birthday",
    ]
    return dataclasses.replace(row, note=random.choice(custom_notes))


class ExpenseAddEmergencyRepair(_ExpenseAddMultiple):
  """Task to add an urgent Housing expense."""

  complexity = 1.3
  n_rows = 1
  n_rows_noise = 5

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    housing_id = next(
        k
        for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
        if v == 'Housing'
    )
    row = _generate_expense(category_id=housing_id)
    return dataclasses.replace(row, name="Emergency Pipe Repair", note="Urgent")


class ExpenseDeleteHighValueExpenses(_ExpenseDeleteMultiple):
  """Task to delete expenses that are over a certain amount (e.g., > $100)."""

  complexity = 2.2
  n_rows = 2
  n_rows_noise = 20

  @property
  def goal(self) -> str:
    return (
        f'In {_APP_NAME}, delete all expenses that are greater than $100.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    expenses = []
    # Generate a mix of expenses
    while len(expenses) < cls.n_rows + cls.n_rows_noise:
      candidate = _generate_expense()
      if not any([candidate.name == expense.name for expense in expenses]):
        expenses.append(candidate)

    # Identify high value ones (> 10000 cents)
    high_value = [e for e in expenses if e.amount > 10000]
    low_value = [e for e in expenses if e.amount <= 10000]

    # Ensure we have enough high value ones to delete
    if len(high_value) < cls.n_rows:
        # Force create some if random generation didn't yield enough
        needed = cls.n_rows - len(high_value)
        for _ in range(needed):
            row = _generate_expense()
            row = dataclasses.replace(row, amount=random.randint(10100, 50000))
            high_value.append(row)
    
    target_rows = high_value[:cls.n_rows]
    # The noise is everything else, plus the unused high value ones
    noise_rows = low_value + high_value[cls.n_rows:]
    
    return {
        sqlite_validators.ROW_OBJECTS: target_rows,
        sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    }


class ExpenseDeleteHousingCategory(_ExpenseDeleteMultiple):
  """Task to delete all expenses belonging to the Housing category."""

  complexity = 2.0
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    return f'Delete all expenses in the Housing category from {_APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    housing_id = next(
        k
        for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
        if v == 'Housing'
    )
    
    expenses = []
    # Generate general expenses
    for _ in range(cls.n_rows_noise):
        expenses.append(_generate_expense())
        
    # Generate target Housing expenses
    targets = []
    for _ in range(cls.n_rows):
        targets.append(_generate_expense(category_id=housing_id))
        
    # Ensure noise doesn't accidentally contain Housing
    final_noise = [e for e in expenses if e.category_id != housing_id]
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: final_noise,
    }


class ExpenseDeleteSpecificNote(_ExpenseDeleteMultiple):
  """Task to delete expenses that contain a specific text in their note."""

  complexity = 2.5
  n_rows = 2
  n_rows_noise = 10
  _TARGET_NOTE = "Duplicate entry"

  @property
  def goal(self) -> str:
    return (
        f'In {_APP_NAME}, remove all expenses that have the note'
        f' "{self._TARGET_NOTE}".'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_expense()
        targets.append(dataclasses.replace(row, note=cls._TARGET_NOTE))
        
    noise = []
    for _ in range(cls.n_rows_noise):
        row = _generate_expense()
        if row.note == cls._TARGET_NOTE:
            row = dataclasses.replace(row, note="Normal note")
        noise.append(row)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteSmallExpenses(_ExpenseDeleteMultiple):
  """Task to delete very small expenses (under $5)."""

  complexity = 2.0
  n_rows = 3
  n_rows_noise = 15

  @property
  def goal(self) -> str:
    return f'Delete any expenses under $5.00 from {_APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = []
    for _ in range(cls.n_rows):
        row = _generate_expense()
        # Set amount < 500 cents
        targets.append(dataclasses.replace(row, amount=random.randint(50, 499)))
        
    noise = []
    for _ in range(cls.n_rows_noise):
        row = _generate_expense()
        # Ensure noise is > 500 cents
        if row.amount < 500:
            row = dataclasses.replace(row, amount=random.randint(600, 5000))
        noise.append(row)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseAddMonthlySubscriptions(_ExpenseAddMultiple):
  """Task to add a set of monthly subscription expenses."""

  complexity = 3.0
  n_rows = 3
  n_rows_noise = 5

  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    others_id = next(
        k
        for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
        if v == 'Others'
    )
    row = _generate_expense(category_id=others_id)
    subs = ['Netflix', 'Spotify', 'Gym', 'AWS', 'Google One', 'Internet']
    return dataclasses.replace(
        row, 
        name=random.choice(subs), 
        note="Monthly recurring"
    )

  @property
  def goal(self) -> str:
    # Override goal to sound more specific about subscriptions
    text_repr = _get_expense_rows_as_text(
        self.params[sqlite_validators.ROW_OBJECTS],
        self.params[_TEXT_REPRESENTATION_TYPE],
    )
    return (
        f'Add the following monthly subscriptions to {_APP_NAME}:\n{text_repr}'
    )