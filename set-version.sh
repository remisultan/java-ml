#!/bin/sh

VERSION_TYPE=$1

APP_VERSION=$(grep -rE "version.next=[0-9]+\.[0-9]+\.[0-9]+" version.properties | cut -d"=" -f2)

if [ "$VERSION_TYPE" = 'release' ]; then
 #Sets release version to maven and updates to minor version in version.properties
 echo "[INFO] Setting project version to $APP_VERSION"
 ./mvnw versions:set -DnewVersion=$APP_VERSION
 LAST_NUMBER=$(echo "$APP_VERSION" | cut -d'.' -f3)
 NEW_VERSION=$(echo "$APP_VERSION" | cut -d'.' -f1,2)\.$(($LAST_NUMBER + 1))
 sed -ri "s#version.next=(.+)#version.next=$NEW_VERSION#g" version.properties
 echo "[INFO] next version is $NEW_VERSION"
elif [ "$VERSION_TYPE" = 'snapshot' ]; then
 #Sets version.properties snapshot version to maven version
 echo "Setting project version to $APP_VERSION-SNAPSHOT"
 ./mvnw versions:set -DnewVersion=$APP_VERSION-SNAPSHOT
 echo "[INFO] Version $APP_VERSION-SNAPSHOT set with success"
else
 echo "VERSION_TYPE must be 'snapshot' or 'release'"
 exit 1
fi

exit 0
