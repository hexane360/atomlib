{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Function Summary') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if modules %}
.. rubric:: {{ _('Sub-modules') }}

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: {{ _('Attributes') }}

{% for item in attributes %}
.. autoattribute:: {{ item }}

{%- endfor %}
{% endif %}

{% if classes %}
.. rubric:: {{ _('Classes') }}

{% for item in classes %}
.. autoclass:: {{ item }}

{%- endfor %}
{% endif %}

{% if exceptions %}
.. rubric:: {{ _('Exceptions') }}

{% for item in exceptions %}
.. autoclass:: {{ item }}

{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: {{ _('Functions') }}

{% for item in functions %}
.. autofunction:: {{ item }}

{%- endfor %}
{% endif %}
