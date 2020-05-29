



class configuration:

```yaml
used_classes:
  - Arson
  - Fight
  - Vandalism
  - Explosion
  - Arrest
  - Robbery
  - Assault
  - Shooting

class_map:
  Fire:
    - Arson
    - Explosion
  Property_damage:
    - Robbery
    - Vandalism
  Disarmed_attack:
    - Assault
    - Fight
  Arrest:
    - Arrest
  Shooting:
    - Shooting
```