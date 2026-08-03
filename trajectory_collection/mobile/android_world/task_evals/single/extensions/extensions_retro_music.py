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

"""Additional tasks for Retro Music app."""

import random
from typing import Any

from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import retro_music
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


class RetroCreatePlaylistFromArtist(retro_music.RetroCreatePlaylist):
  """Task to create a playlist containing all songs by a specific artist."""

  complexity = 3.0

  @property
  def goal(self) -> str:
    return (
        f'Create a playlist in Retro Music titled "{self.params["playlist_name"]}"'
        f' containing all songs by the artist "{self.params["artist_name"]}".'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    # Bypass RetroCreatePlaylist.initialize_task to perform custom file generation
    super(retro_music.RetroCreatePlaylist, self).initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    retro_music._clear_playlist_dbs(env)

    # Write target songs with the specific artist
    for file in self.params['files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=self.params['artist_name'],
          duration_milliseconds=random.randint(180000, 300000),
      )

    # Write noise songs with random artists (ensuring not the target artist)
    for file in self.params['noise_files']:
      noise_artist = random.choice(user_data_generation.COMMON_GIVEN_NAMES)
      while noise_artist == self.params['artist_name']:
        noise_artist = random.choice(user_data_generation.COMMON_GIVEN_NAMES)
      
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=noise_artist,
          duration_milliseconds=random.randint(180000, 300000),
      )
    
    retro_music._scan_music_directory(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    base_params = retro_music.RetroCreatePlaylist.generate_random_params()
    base_params['artist_name'] = random.choice(
        user_data_generation.COMMON_GIVEN_NAMES
    )
    return base_params


class RetroCreateTwoPlaylists(retro_music.RetroCreatePlaylist):
  """Task to create two different playlists."""

  complexity = 3.5

  @property
  def goal(self) -> str:
    p1_songs = ', '.join(f.split('.')[0] for f in self.params['files'])
    p2_songs = ', '.join(f.split('.')[0] for f in self.params['files2'])
    return (
        f'Create a playlist titled "{self.params["playlist_name"]}" with: {p1_songs}. '
        f'Then create another playlist titled "{self.params["playlist_name2"]}" '
        f'with: {p2_songs}.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    # Manually write all files for both playlists + noise
    all_files = (
        self.params['files'] + self.params['files2'] + self.params['noise_files']
    )
    super(retro_music.RetroCreatePlaylist, self).initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    retro_music._clear_playlist_dbs(env)

    for file in all_files:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(180000, 300000),
      )
    retro_music._scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    p1_success = sqlite_validators.verify_playlist(
        actual,
        self.params['playlist_name'],
        [f.split('.')[0] for f in self.params['files']],
    )
    p2_success = sqlite_validators.verify_playlist(
        actual,
        self.params['playlist_name2'],
        [f.split('.')[0] for f in self.params['files2']],
    )
    return (int(p1_success) + int(p2_success)) / 2.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    params2 = retro_music.RetroCreatePlaylist.generate_random_params()
    params['playlist_name2'] = params2['playlist_name']
    params['files2'] = params2['files']
    # Ensure distinct playlist names
    if params['playlist_name'] == params['playlist_name2']:
        params['playlist_name2'] += " 2"
    return params


class RetroQueueArtist(RetroCreatePlaylistFromArtist):
  """Task to add all songs by a specific artist to the playing queue."""

  complexity = 3.2

  @property
  def goal(self) -> str:
    return (
        f'Add all songs by the artist "{self.params["artist_name"]}" to the '
        'playing queue in Retro Music.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    queue = retro_music._get_playing_queue(env)
    expected = sorted([f.split('.')[0] for f in self.params['files']])
    actual = sorted(queue)
    # Strict check: Queue must contain exactly these songs
    return 1.0 if set(queue) == set(expected) else 0.0


class RetroCreatePlaylistExcludeSong(retro_music.RetroCreatePlaylist):
  """Task to create a playlist with all songs except one specific song."""

  complexity = 2.5

  @property
  def goal(self) -> str:
    excluded = self.params['excluded_song'].split('.')[0]
    return (
        f'Create a playlist titled "{self.params["playlist_name"]}" containing '
        f'all available songs except "{excluded}".'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    # If no noise files were generated, move one target file to noise to be the excluded one
    if not params['noise_files']:
        if params['files']:
             params['noise_files'] = [params['files'].pop()]
        else:
             # Fallback just in case
             params['noise_files'] = ["ExcludedSong.mp3"]

    params['excluded_song'] = params['noise_files'][0]
    return params


class RetroQueueSpecificOrder(retro_music.RetroPlayingQueue):
  """Task to queue songs in a specific, randomized order."""

  complexity = 3.5

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    # Shuffle the target files to create a requirement for specific ordering
    random.shuffle(params['files'])
    return params


class RetroCreateFavoritesPlaylist(retro_music.RetroCreatePlaylist):
  """Task to create a 'Favorites' playlist."""

  complexity = 2.0

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Create a playlist named "Favorites" containing the following songs: {names}'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    params['playlist_name'] = 'Favorites'
    return params


class RetroCreatePlaylistShortSongs(retro_music.RetroCreatePlaylist):
  """Task to create a playlist with songs shorter than 3 minutes."""
  
  complexity = 3.0
  
  @property
  def goal(self) -> str:
      return (
          f'Create a playlist named "{self.params["playlist_name"]}" containing '
          'all songs that are shorter than 3 minutes.'
      )

  def initialize_task(self, env: interface.AsyncEnv):
    super(retro_music.RetroCreatePlaylist, self).initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    retro_music._clear_playlist_dbs(env)

    # Write target songs (short)
    for file in self.params['files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          duration_milliseconds=random.randint(60000, 179000), # < 3 mins
      )

    # Write noise songs (long)
    for file in self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          duration_milliseconds=random.randint(181000, 300000), # > 3 mins
      )
    retro_music._scan_music_directory(env)


class RetroDuplicatePlaylist(retro_music.RetroCreatePlaylist):
  """Task to create two playlists with the exact same content."""

  complexity = 3.0

  @property
  def goal(self) -> str:
    songs = ', '.join(f.split('.')[0] for f in self.params['files'])
    return (
        f'Create a playlist named "{self.params["playlist_name"]}" with songs: {songs}. '
        f'Then create a second playlist named "{self.params["playlist_name_2"]}" '
        'with the exact same songs.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = retro_music._get_playlist_data(env)
    expected_songs = [f.split('.')[0] for f in self.params['files']]
    p1 = sqlite_validators.verify_playlist(
        actual, self.params['playlist_name'], expected_songs
    )
    p2 = sqlite_validators.verify_playlist(
        actual, self.params['playlist_name_2'], expected_songs
    )
    return (int(p1) + int(p2)) / 2.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    params['playlist_name_2'] = params['playlist_name'] + " Copy"
    return params


class RetroPlaySingleSong(retro_music.RetroPlayingQueue):
  """Task to play (or queue) a single specific song."""

  complexity = 1.5

  @property
  def goal(self) -> str:
    return f'Play the song "{self.params["files"][0].split(".")[0]}" in Retro Music.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Generate standard params but restrict 'files' to 1 to simplify the goal
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    # Move extra target files to noise to clean up the environment
    params['noise_files'] += params['files'][1:] 
    params['files'] = [params['files'][0]]
    return params


class RetroCreatePlaylistReverseOrder(retro_music.RetroCreatePlaylist):
  """Task to create a playlist with songs sorted in reverse alphabetical order."""

  complexity = 2.8

  @property
  def goal(self) -> str:
    return (
        f'Create a playlist named "{self.params["playlist_name"]}" containing '
        'all available songs sorted in reverse alphabetical order.'
    )

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    params = retro_music.RetroCreatePlaylist.generate_random_params()
    all_songs = params['files'] + params['noise_files']
    # The user is expected to use all songs, sorted Z-A
    sorted_songs = sorted(all_songs, reverse=True)
    
    params['files'] = sorted_songs
    params['noise_files'] = [] # No noise, all songs are targets
    return params