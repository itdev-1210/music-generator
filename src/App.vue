<template>
  <!-- <v-app>
    <v-app-bar app>
      <v-toolbar-title class="headline text-uppercase">
        <span>Vuetify</span>
        <span class="font-weight-light">MATERIAL DESIGN</span>
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn
        text
        href="https://github.com/vuetifyjs/vuetify/releases/latest"
        target="_blank"
      >
        <span class="mr-2">Latest Release</span>
      </v-btn>
    </v-app-bar>


    <v-content>
      <SimpleUpload/>
    </v-content>

  </v-app> -->
  <div class="section">
    <div class="container">

      <simple-upload />

    </div>
  </div>
</template>

<script>
/* eslint-disable */
import HelloWorld from './components/HelloWorld';
import SimpleUpload from './components/SimpleUpload';

export default {
  name: 'App',
  components: {
    HelloWorld,
    SimpleUpload
  },

  computed: {
    isDisabled() {
      const fileUploaded = this.canGen

      console.log('comp')


      return fileUploaded
    }
  },

  data: () => ({
    file: "",
    msg: "",
    error: false,
    uploading: false,
    canGen: false
  })
};
</script>
